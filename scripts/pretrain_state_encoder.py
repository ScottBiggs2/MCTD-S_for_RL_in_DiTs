"""
Pretraining script for StateCNNEncoder with reconstruction and auxiliary tasks.

Stage 1 of hybrid training approach:
- Train state encoder with weighted reconstruction loss
- Predict agent position, goal position, and direction
- Generate visualizations of true vs reconstructed grids
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional
import gymnasium as gym
import random
import torch.nn.functional as F

# Import minigrid to register environments
try:
    import minigrid
except ImportError:
    print("ERROR: minigrid package not found. Install with: pip install minigrid")
    sys.exit(1)

from src.models.state_cnn_encoder import StateCNNEncoder
from src.training.state_encoder_pretrainer import StateEncoderPretrainer
from src.config import get_experiment_config
from src.environments.minigrid_wrapper import get_full_grid_image


def extract_positions_from_grid(grid: np.ndarray, env: gym.Env) -> tuple:
    """
    Extract agent and goal positions from environment.
    
    Args:
        grid: Grid image (H, W, 3) or None (will extract from env)
        env: Gymnasium environment
    
    Returns:
        (agent_pos, goal_pos): Both are (x, y) tuples or None
    """
    unwrapped = env.unwrapped
    
    # Get agent position
    agent_pos = None
    if hasattr(unwrapped, 'agent_pos'):
        agent_pos = tuple(unwrapped.agent_pos)  # (x, y)
    
    # Get goal position by searching grid
    goal_pos = None
    if hasattr(unwrapped, 'grid'):
        grid_obj = unwrapped.grid
        for x in range(grid_obj.width):
            for y in range(grid_obj.height):
                try:
                    cell = grid_obj.get(x, y)
                    if cell is not None and hasattr(cell, 'type'):
                        if cell.type == 'goal':
                            goal_pos = (x, y)
                            break
                except:
                    continue
        if goal_pos:
            return agent_pos, goal_pos
    
    return agent_pos, goal_pos


# Import augmentation function from shared utility
from src.utils.data_augmentation import apply_grid_augmentation


class PretrainingDataset:
    """
    Dataset for state encoder pretraining.
    
    Extracts agent and goal positions from trajectories.
    Supports data augmentation via rotation and reflection.
    """
    def __init__(
        self,
        trajectories: list,
        env_name: str,
        grid_size: int = 19,
        use_augmentation: bool = True,
    ):
        self.trajectories = trajectories
        self.env_name = env_name
        self.grid_size = grid_size
        self.use_augmentation = use_augmentation
        
        # Pre-extract positions for all trajectories
        print("Extracting agent and goal positions from trajectories...")
        self.data = []
        
        for traj_idx, traj in enumerate(trajectories):
            if traj_idx % 100 == 0:
                print(f"  Processing trajectory {traj_idx}/{len(trajectories)}")
            
            # Create environment to extract positions
            env = gym.make(env_name)
            episode_seed = traj.get('episode_seed')
            
            if episode_seed is not None:
                obs, _ = env.reset(seed=int(episode_seed))
            else:
                obs, _ = env.reset()
            
            # Extract grid directly from environment to ensure consistency
            # This ensures we're using the actual environment state, not stored data
            from src.environments.minigrid_wrapper import get_full_grid_image
            grid_image = get_full_grid_image(env)  # (H, W, 3) uint8 array with indices
            
            # Extract positions
            agent_pos, goal_pos = extract_positions_from_grid(None, env)
            
            # Convert grid to tensor - grid_image is already (H, W, 3) with uint8 indices
            # Values are in range [0, 11] typically (object type, color, state indices)
            grid_tensor = torch.tensor(grid_image, dtype=torch.float32)  # Keep as float32 for model
            
            # Verify grid shape
            if grid_tensor.shape != (grid_size, grid_size, 3):
                print(f"Warning: Grid shape mismatch. Expected ({grid_size}, {grid_size}, 3), got {grid_tensor.shape}")
                env.close()
                continue
            
            # Get direction from environment
            direction = env.unwrapped.agent_dir
            direction_tensor = torch.tensor(direction, dtype=torch.long)
            
            # Convert positions to tensors
            agent_pos_tensor = None
            goal_pos_tensor = None
            if agent_pos is not None:
                agent_pos_tensor = torch.tensor(agent_pos, dtype=torch.long)
            if goal_pos is not None:
                goal_pos_tensor = torch.tensor(goal_pos, dtype=torch.long)
            
            self.data.append({
                'grid': grid_tensor,
                'direction': direction_tensor,
                'agent_pos': agent_pos_tensor,
                'goal_pos': goal_pos_tensor,
            })
            
            env.close()
        
        print(f"Processed {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Apply random augmentation if enabled
        if self.use_augmentation and random.random() < 0.75:  # 75% chance of augmentation
            aug_types = ['rot90', 'rot180', 'rot270', 'hflip', 'vflip']
            aug_type = random.choice(aug_types)
            
            grid_aug, agent_pos_aug, goal_pos_aug, direction_aug = apply_grid_augmentation(
                sample['grid'],
                sample['agent_pos'],
                sample['goal_pos'],
                sample['direction'],
                self.grid_size,
                aug_type
            )
            
            return {
                'grid': grid_aug,
                'direction': direction_aug,
                'agent_pos': agent_pos_aug,
                'goal_pos': goal_pos_aug,
            }
        else:
            return sample


def collate_fn(batch):
    """Custom collate function to handle optional agent_pos and goal_pos."""
    grids = torch.stack([item['grid'] for item in batch])
    directions = torch.stack([item['direction'] for item in batch])
    
    # Handle optional positions
    agent_positions = [item['agent_pos'] for item in batch]
    goal_positions = [item['goal_pos'] for item in batch]
    
    # Check if all have positions
    has_agent_pos = all(p is not None for p in agent_positions)
    has_goal_pos = all(p is not None for p in goal_positions)
    
    result = {
        'grid': grids,
        'direction': directions,
    }
    
    if has_agent_pos:
        result['agent_pos'] = torch.stack(agent_positions)
    if has_goal_pos:
        result['goal_pos'] = torch.stack(goal_positions)
    
    return result


def main():
    """Main pretraining function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pretrain StateCNNEncoder')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data pickle file')
    parser.add_argument('--env', type=str, required=True,
                       help='Environment name (e.g., MiniGrid-FourRooms-v0)')
    parser.add_argument('--grid_size', type=int, default=19,
                       help='Grid size (default: 19 for FourRooms)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Hidden dimension (default: use config default)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/pretrained_state_encoder',
                       help='Checkpoint directory')
    parser.add_argument('--viz_dir', type=str, default='outputs/state_encoder_viz',
                       help='Visualization directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu, cuda, mps)')
    
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
    
    # Load config to get defaults
    from src.config import get_experiment_config
    experiment_config = get_experiment_config()
    
    # Use config default if not specified
    if args.hidden_dim is None:
        args.hidden_dim = experiment_config.model.hidden_dim
        print(f"Using hidden_dim from config: {args.hidden_dim}")
    else:
        print(f"Using hidden_dim from command line: {args.hidden_dim}")
    
    # Load trajectories
    print(f"Loading trajectories from {args.data}...")
    with open(args.data, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Create dataset with augmentation
    print("Creating pretraining dataset...")
    dataset = PretrainingDataset(
        trajectories=trajectories,
        env_name=args.env,
        grid_size=args.grid_size,
        use_augmentation=True,  # Enable augmentation for 4x effective data
    )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Model
    print("\nInitializing StateCNNEncoder...")
    model = StateCNNEncoder(
        grid_size=args.grid_size,
        hidden_dim=args.hidden_dim,
        num_channels=3,
        enable_decoder=True,
        enable_auxiliary=True,
    )
    
    # Config
    config = {
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'lr_patience': 5,
        'grid_size': args.grid_size,
        'hidden_dim': args.hidden_dim,  # Store in config for checkpoint
        # Concept: Learn the hard things first (ie, the agent and goal positions), then easier things
        'reconstruction_weight': 0.25,
        'agent_pos_weight': 1,  # Reduced from 1.0
        'goal_pos_weight': 1,
        'direction_weight': 0.5,
        # No KL weight needed - VAE removed
    }
    
    # Trainer
    print("\nInitializing trainer...")
    trainer = StateEncoderPretrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    
    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.checkpoint_dir = checkpoint_dir
    
    viz_dir = Path(args.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    print(f"\n{'='*60}")
    print(f"Pretraining StateCNNEncoder for {args.num_epochs} epochs")
    print(f"{'='*60}\n")
    
    trainer.train(
        num_epochs=args.num_epochs,
        logger=None,
        viz_dir=viz_dir,
    )
    
    print("\nPretraining complete!")
    print(f"Best model saved to: {checkpoint_dir}/")
    print(f"Visualizations saved to: {viz_dir}/")


if __name__ == "__main__":
    main()
