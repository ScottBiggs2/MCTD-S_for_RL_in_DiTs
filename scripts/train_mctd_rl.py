"""
RL Training script using MCTD search.

Fuses train_baseline.py and test_mctd_search.py to:
1. Load a pretrained model checkpoint
2. Use MCTD to collect trajectories (RL)
3. Train on collected trajectories using masked diffusion
4. Save checkpoints in separate subfolder

IMPORTANT: Collects ALL valid trajectories (both successful and failed).
The model needs to learn from failures to improve. Only trajectories with
invalid actions or empty sequences are skipped.

Example usage:
    python scripts/train_mctd_rl.py \
        --checkpoint checkpoints/best_model_epoch8.pt \
        --env MiniGrid-FourRooms-v0 \
        --rl_iterations 10 \
        --rl_episodes_per_iter 50 \
        --num_epochs_per_iter 5 \
        --num_simulations 250 \
        --use_distance_reward \
        --device mps
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import DataLoader
from collections import defaultdict
import argparse
import gymnasium as gym

# Import minigrid to register environments
try:
    import minigrid
except ImportError:
    print("ERROR: minigrid package not found. Install with: pip install minigrid")
    sys.exit(1)

from src.models.diffusion_policy import DiffusionPolicy
from src.models.action_encoder import ActionEncoder
from src.mctd import HiddenSpaceMCTD
from src.environments.trajectory_dataset import TrajectoryDataset
from src.training.mdlm_trainer import MaskedDiffusionTrainer
from src.utils.logging import Logger
from src.utils.checkpointing import save_checkpoint
from src.config import get_experiment_config, get_model_config, get_mctd_config, load_config_from_dict
from src.environments.minigrid_wrapper import get_full_grid_image


def load_model(checkpoint_path: str, device: str = 'cpu') -> DiffusionPolicy:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load config from checkpoint, otherwise use default
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        experiment_config = load_config_from_dict(checkpoint['config'])
        model_config_dict = experiment_config.to_dict()
        model_config = {
            'num_actions': model_config_dict['num_actions'],
            'hidden_dim': model_config_dict['hidden_dim'],
            'num_layers': model_config_dict['num_layers'],
            'num_heads': model_config_dict['num_heads'],
            'num_tokens': model_config_dict['num_tokens'],
            'max_seq_len': model_config_dict['max_seq_len'],
            'dropout': model_config_dict['dropout'],
        }
    else:
        model_config_obj = get_model_config()
        model_config = {
            'num_actions': model_config_obj.num_actions,
            'hidden_dim': model_config_obj.hidden_dim,
            'num_layers': model_config_obj.num_layers,
            'num_heads': model_config_obj.num_heads,
            'num_tokens': model_config_obj.num_tokens,
            'max_seq_len': model_config_obj.max_seq_len,
            'dropout': model_config_obj.dropout,
        }
    
    model = DiffusionPolicy(**model_config)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'config' not in checkpoint:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    return model.to(device)


def collect_mctd_trajectory(
    model: DiffusionPolicy,
    action_encoder: ActionEncoder,
    env: gym.Env,
    env_name: str,
    mctd: HiddenSpaceMCTD,
    max_steps: int = 64,
    seed: Optional[int] = None,
    use_distance_reward: bool = False,
    distance_reward_scale: float = 0.1,
) -> Tuple[Dict, bool]:
    """
    Collect a single trajectory using MCTD search.
    
    Returns:
        trajectory: dict with 'states' and 'actions'
        success: whether goal was reached
    """
    # Reset environment
    if seed is not None:
        obs, info = env.reset(seed=int(seed))
    else:
        obs, info = env.reset()
    
    # Get initial state
    full_grid_image = get_full_grid_image(env)
    initial_state = {
        'grid': full_grid_image,
        'direction': obs['direction'],
    }
    
    # Run MCTD search
    mctd_actions, _ = mctd.search(
        initial_state,
        reference_path=None,
        use_similarity_reward=False,
        use_distance_reward=use_distance_reward,
        distance_reward_scale=distance_reward_scale,
    )
    
    # Execute actions and collect trajectory
    states = []
    actions = []
    current_state = initial_state.copy()
    
    success = False
    steps = 0
    
    for action in mctd_actions:
        if steps >= max_steps:
            break
        
        # Store state before action (state-action pair)
        states.append({
            'grid': current_state['grid'].copy(),
            'direction': current_state['direction'],
        })
        actions.append(int(action))
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        steps += 1
        
        # Update current state for next iteration
        current_state = {
            'grid': get_full_grid_image(env),
            'direction': obs['direction'],
        }
        
        if done:
            if reward > 0.1:  # Success
                success = True
            break
    
    # Note: states and actions should have same length (state before each action)
    # The dataset will handle padding/truncation as needed
    
    trajectory = {
        'states': states,
        'actions': actions,
    }
    
    return trajectory, success


def collect_rl_episodes(
    model: DiffusionPolicy,
    action_encoder: ActionEncoder,
    env_name: str,
    num_episodes: int,
    mctd_config: Dict,
    device: str = 'cpu',
    base_seed: int = 42,
    use_distance_reward: bool = False,
    distance_reward_scale: float = 0.1,
) -> List[Dict]:
    """
    Collect trajectories using MCTD search (RL data collection).
    
    Collects ALL valid trajectories (both successful and failed), not just successful ones.
    The model needs to learn from failures to improve. Only trajectories with invalid
    actions or empty sequences are skipped.
    
    Returns:
        List of trajectories (dicts with 'states' and 'actions')
    """
    trajectories = []
    env = gym.make(env_name)
    
    # Create MCTD searcher
    mctd = HiddenSpaceMCTD(
        policy_model=model,
        env=env,
        action_encoder=action_encoder,
        num_simulations=mctd_config['num_simulations'],
        exploration_const=mctd_config['exploration_const'],
        guidance_scales=mctd_config['guidance_scales'],
        sparse_timesteps=mctd_config.get('sparse_timesteps', None),
        denoising_step_size=mctd_config['denoising_step_size'],
        reward_alpha=mctd_config.get('reward_alpha', 0.1),
        device=device,
    )
    
    success_count = 0
    
    for episode in range(num_episodes):
        episode_seed = base_seed + episode if base_seed is not None else None
        
        trajectory, success = collect_mctd_trajectory(
            model=model,
            action_encoder=action_encoder,
            env=env,
            env_name=env_name,
            mctd=mctd,
            max_steps=64,
            seed=episode_seed,
            use_distance_reward=use_distance_reward,
            distance_reward_scale=distance_reward_scale,
        )
        
        # Collect ALL well-formed trajectories (not just successful ones)
        # The model needs to learn from both successes and failures
        # Only skip if trajectory is empty or has invalid actions
        if len(trajectory['actions']) > 0:
            # Validate actions are in valid range (0-6 for MiniGrid)
            actions_valid = all(0 <= a < 7 for a in trajectory['actions'])
            if actions_valid:
                trajectories.append(trajectory)
                if success:
                    success_count += 1
                    print(f"  Episode {episode+1}/{num_episodes}: SUCCESS (collected {len(trajectories)} trajectories)")
                else:
                    print(f"  Episode {episode+1}/{num_episodes}: FAILED (collected {len(trajectories)} trajectories)")
            else:
                print(f"  Episode {episode+1}/{num_episodes}: INVALID ACTIONS (skipped)")
        else:
            print(f"  Episode {episode+1}/{num_episodes}: EMPTY TRAJECTORY (skipped)")
    
    env.close()
    
    print(f"\nCollected {len(trajectories)} trajectories out of {num_episodes} episodes")
    print(f"Success rate: {success_count/num_episodes*100:.1f}% ({success_count} successful, {len(trajectories)-success_count} failed but valid)")
    
    return trajectories


def main():
    parser = argparse.ArgumentParser(description='RL training with MCTD search')
    
    # Model checkpoint (required)
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to pretrained model checkpoint')
    
    # Environment
    parser.add_argument('--env', type=str, default='MiniGrid-FourRooms-v0',
                       help='MiniGrid environment name')
    
    # RL collection parameters
    parser.add_argument('--rl_episodes_per_iter', type=int, default=50,
                       help='Number of MCTD episodes to collect per RL iteration')
    parser.add_argument('--rl_iterations', type=int, default=10,
                       help='Number of RL training iterations')
    parser.add_argument('--min_trajectories_per_train', type=int, default=10,
                       help='Minimum successful trajectories needed before training')
    
    # MCTD search parameters (from test_mctd_search.py)
    parser.add_argument('--num_simulations', type=int, default=None,
                       help='Number of MCTD simulations (overrides config default)')
    parser.add_argument('--exploration_const', type=float, default=None,
                       help='UCT exploration constant (overrides config default)')
    parser.add_argument('--guidance_scales', type=float, nargs='+', default=None,
                       help='Guidance scales for expansion, e.g., --guidance_scales 0.0 0.5 1.0')
    parser.add_argument('--denoising_step_size', type=float, default=None,
                       help='Denoising step size (overrides config default)')
    parser.add_argument('--use_distance_reward', action='store_true',
                       help='Enable intermediate rewards based on Manhattan distance to goal')
    parser.add_argument('--distance_reward_scale', type=float, default=0.1,
                       help='Scale factor for distance-based rewards (default 0.1)')
    
    # Training parameters (from train_baseline.py)
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Training batch size (overrides config default)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config default)')
    parser.add_argument('--num_epochs_per_iter', type=int, default=5,
                       help='Number of training epochs per RL iteration')
    
    # Device and paths
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Base checkpoint directory')
    parser.add_argument('--rl_checkpoint_subdir', type=str, default='mctd_rl',
                       help='Subdirectory for RL checkpoints (inside checkpoint_dir)')
    parser.add_argument('--base_seed', type=int, default=42,
                       help='Base seed for environment episodes')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load experiment config
    experiment_config = get_experiment_config()
    config = experiment_config.to_dict()
    
    # Override config with command-line args
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    
    # Load pretrained model
    print(f"\nLoading pretrained model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=device)
    model.train()  # Set to training mode for RL updates
    
    # Create action encoder
    print("Creating action encoder...")
    action_encoder = ActionEncoder(
        num_actions=config['num_actions'],
        hidden_dim=config['hidden_dim'],
    ).to(device)
    
    # Setup MCTD config
    mctd_config_obj = get_mctd_config()
    mctd_config = {
        'num_simulations': args.num_simulations if args.num_simulations is not None else mctd_config_obj.num_simulations,
        'exploration_const': args.exploration_const if args.exploration_const is not None else mctd_config_obj.exploration_const,
        'guidance_scales': args.guidance_scales if args.guidance_scales is not None else mctd_config_obj.guidance_scales,
        'denoising_step_size': args.denoising_step_size if args.denoising_step_size is not None else mctd_config_obj.denoising_step_size,
        'sparse_timesteps': mctd_config_obj.sparse_timesteps,
        'reward_alpha': mctd_config_obj.reward_alpha,
    }
    
    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.rl_checkpoint_subdir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"RL checkpoints will be saved to: {checkpoint_dir}")
    
    # Logger
    logger = None
    if args.use_wandb:
        logger = Logger(use_wandb=True, project_name="maze-mctd-rl")
    
    # RL training loop
    print(f"\n{'='*60}")
    print(f"Starting RL training with MCTD")
    print(f"  RL iterations: {args.rl_iterations}")
    print(f"  Episodes per iteration: {args.rl_episodes_per_iter}")
    print(f"  Training epochs per iteration: {args.num_epochs_per_iter}")
    print(f"{'='*60}\n")
    
    all_trajectories = []  # Accumulate trajectories across iterations
    trainer = None  # Will be initialized in first iteration
    
    for rl_iter in range(args.rl_iterations):
        print(f"\n{'='*60}")
        print(f"RL Iteration {rl_iter+1}/{args.rl_iterations}")
        print(f"{'='*60}")
        
        # 1. Collect trajectories using MCTD
        print(f"\nStep 1: Collecting trajectories with MCTD...")
        new_trajectories = collect_rl_episodes(
            model=model,
            action_encoder=action_encoder,
            env_name=args.env,
            num_episodes=args.rl_episodes_per_iter,
            mctd_config=mctd_config,
            device=device,
            base_seed=args.base_seed + rl_iter * args.rl_episodes_per_iter,
            use_distance_reward=args.use_distance_reward,
            distance_reward_scale=args.distance_reward_scale,
        )
        
        # Add to accumulated trajectories
        all_trajectories.extend(new_trajectories)
        
        # Check if we have enough trajectories to train
        # Note: We now collect all valid trajectories (successful + failed), so this threshold
        # should be easier to meet. But we still want some minimum to ensure quality.
        if len(all_trajectories) < args.min_trajectories_per_train:
            print(f"\n⚠️  Only {len(all_trajectories)} trajectories collected (need {args.min_trajectories_per_train})")
            print("Skipping training this iteration...")
            continue
        
        # 2. Create dataset from collected trajectories
        print(f"\nStep 2: Creating dataset from {len(all_trajectories)} trajectories...")
        dataset = TrajectoryDataset(all_trajectories, max_seq_len=config['max_seq_len'])
        train_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        
        # Create dummy val_loader (we don't have validation data in RL)
        val_loader = DataLoader(
            dataset,  # Use same data for validation (just for compatibility)
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        
        # 3. Train on collected trajectories
        print(f"\nStep 3: Training on collected trajectories...")
        # Create new trainer (or reuse if exists) - this ensures optimizer state is fresh
        trainer = MaskedDiffusionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )
        
        trainer.checkpoint_dir = checkpoint_dir
        trainer.experiment_config = experiment_config
        
        # Train for a few epochs
        trainer.train(
            num_epochs=args.num_epochs_per_iter,
            logger=logger,
        )
        
        # 4. Save checkpoint
        checkpoint_filename = f"rl_iter_{rl_iter+1}_model.pt"
        metrics = {
            'loss': trainer.best_val_loss if hasattr(trainer, 'best_val_loss') else float('inf'),
            'train_loss': trainer.best_val_loss if hasattr(trainer, 'best_val_loss') else float('inf'),
        }
        save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            epoch=rl_iter+1,
            metrics=metrics,
            checkpoint_dir=checkpoint_dir,
            filename=checkpoint_filename,
            config=experiment_config.to_dict() if hasattr(experiment_config, 'to_dict') else experiment_config,
        )
        print(f"\nSaved checkpoint to: {checkpoint_dir / checkpoint_filename}")
        
        # Log metrics
        if logger:
            logger.log({
                'rl_iteration': rl_iter + 1,
                'trajectories_collected': len(new_trajectories),
                'total_trajectories': len(all_trajectories),
                'success_rate': len(new_trajectories) / args.rl_episodes_per_iter,
            })
    
    # Save final model
    final_metrics = {
        'loss': trainer.best_val_loss if trainer is not None and hasattr(trainer, 'best_val_loss') else float('inf'),
        'total_trajectories': len(all_trajectories),
    }
    
    if trainer is not None:
        save_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            epoch=args.rl_iterations,
            metrics=final_metrics,
            checkpoint_dir=checkpoint_dir,
            filename="final_rl_model.pt",
            config=experiment_config.to_dict() if hasattr(experiment_config, 'to_dict') else experiment_config,
        )
    else:
        # If no training happened, create a dummy optimizer for checkpoint format
        dummy_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        save_checkpoint(
            model=model,
            optimizer=dummy_optimizer,
            epoch=args.rl_iterations,
            metrics=final_metrics,
            checkpoint_dir=checkpoint_dir,
            filename="final_rl_model.pt",
            config=experiment_config.to_dict() if hasattr(experiment_config, 'to_dict') else experiment_config,
        )
    
    final_checkpoint_path = checkpoint_dir / "final_rl_model.pt"
    print(f"\n{'='*60}")
    print(f"RL Training Complete!")
    print(f"Final model saved to: {final_checkpoint_path}")
    print(f"Total trajectories collected: {len(all_trajectories)}")
    print(f"{'='*60}")
    
    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
