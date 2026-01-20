"""
Standalone evaluation script for raw DiT diffusion policy (no MCTD search).

Evaluates the policy directly using greedy decoding and reports:
- Success rate
- Path length statistics
- Action diversity
- Per-sample results with visualizations
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

# Import minigrid to register environments
try:
    import minigrid
except ImportError:
    print("ERROR: minigrid package not found. Install with: pip install minigrid")
    sys.exit(1)

import gymnasium as gym
from src.models.diffusion_policy import DiffusionPolicy
from src.models.action_encoder import ActionEncoder
from src.environments.trajectory_dataset import TrajectoryDataset
from src.config import get_model_config, load_config_from_dict
from src.environments.minigrid_wrapper import get_full_grid_image


def load_model(checkpoint_path: str, device: str = 'cpu') -> DiffusionPolicy:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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
    
    model.eval()
    return model.to(device)


def direct_policy_decode(
    model: DiffusionPolicy, 
    state: Dict[str, torch.Tensor], 
    device: str = 'cpu', 
    num_denoise_steps: int = 20,
    guidance_scale: float = 1.0
) -> torch.Tensor:
    """
    Direct policy decode: greedy action selection without search.
    
    Args:
        model: DiffusionPolicy model
        state: State dict with 'grid' and 'direction'
        device: Device to run on
        num_denoise_steps: Number of denoising steps (more = better quality, slower)
        guidance_scale: Guidance scale for denoising (1.0 = standard)
    
    Returns:
        actions: [seq_len] discrete action sequence
    """
    model.eval()
    with torch.no_grad():
        seq_len = model.max_seq_len
        
        # Initialize from mean action embedding (forward action = 2)
        init_actions = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
        init_hidden = model.action_encoder(init_actions)  # [1, seq_len, hidden_dim]
        
        # Add noise scaled by timestep
        noise = torch.randn_like(init_hidden)
        hidden_state = init_hidden + 0.5 * noise
        
        # Denoise in steps
        for i in range(num_denoise_steps):
            t = 1.0 - (i + 1) / num_denoise_steps
            t_tensor = torch.tensor([t], device=device)
            hidden_state = model.denoise_step(hidden_state, state, t_tensor, guidance_scale=guidance_scale)
        
        # Final decode to actions
        t_final = torch.tensor([0.0], device=device)
        logits = model.forward(hidden_state, state, t_final)
        actions = logits.argmax(dim=-1)[0]  # [seq_len]
        
        # Ensure actions are valid
        actions = torch.clamp(actions, 0, model.num_actions - 1)
        
        return actions.cpu()


def execute_path(
    env, 
    actions: torch.Tensor, 
    seed: Optional[int] = None, 
    max_steps: int = 100
) -> Tuple[List[Tuple[int, int]], bool, int, float]:
    """
    Execute action sequence in environment and return path positions.
    
    Returns:
        positions: List of (x, y) positions
        success: Whether goal was reached
        steps_taken: Number of steps executed
        final_reward: Final reward received
    """
    if seed is not None:
        obs, info = env.reset(seed=int(seed))
    else:
        obs, info = env.reset()
    
    positions = []
    if hasattr(env.unwrapped, 'agent_pos'):
        positions.append(tuple(env.unwrapped.agent_pos))
    
    success = False
    steps = 0
    final_reward = 0.0
    
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        steps += 1
        final_reward = reward
        
        if hasattr(env.unwrapped, 'agent_pos'):
            pos = env.unwrapped.agent_pos
            positions.append(tuple(pos))
        
        if done:
            if reward > 0.1:
                success = True
            break
        
        if steps >= max_steps:
            break
    
    return positions, success, steps, final_reward


def visualize_path(
    env_name: str, 
    predicted_path: List[Tuple[int, int]], 
    optimal_path: Optional[List[Tuple[int, int]]] = None,
    seed: Optional[int] = None, 
    save_path: Optional[str] = None
):
    """Visualize predicted path and optionally optimal path."""
    env = gym.make(env_name)
    if seed is not None:
        obs, info = env.reset(seed=int(seed))
    else:
        obs, info = env.reset()
    
    if hasattr(env.unwrapped, 'grid'):
        grid = env.unwrapped.grid
        width = grid.width
        height = grid.height
    else:
        width = height = 8
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw grid
    for x in range(width):
        for y in range(height):
            try:
                cell = grid.get(x, y)
                if cell and hasattr(cell, 'type'):
                    if cell.type == 'wall':
                        color = 'gray'
                    elif cell.type == 'door':
                        color = 'yellow'
                    elif cell.type == 'goal':
                        color = 'lightgreen'
                    elif cell.type == 'key':
                        color = 'orange'
                    else:
                        color = 'white'
                else:
                    color = 'white'
            except:
                color = 'white'
            
            rect = patches.Rectangle((x, y), 1, 1, linewidth=0.5, 
                                   edgecolor='gray', facecolor=color, alpha=0.3)
            ax.add_patch(rect)
    
    # Plot paths
    if predicted_path and len(predicted_path) >= 2:
        xs, ys = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in predicted_path])
        ax.plot(xs, ys, 'o-', color='red', alpha=0.8, linewidth=2.5,
               label='Predicted Path', markersize=5, zorder=5)
    
    if optimal_path and len(optimal_path) >= 2:
        xs, ys = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in optimal_path])
        ax.plot(xs, ys, 'o-', color='blue', alpha=0.8, linewidth=2.5,
               label='Optimal Path', markersize=5, zorder=5)
    
    # Mark start and goal
    if predicted_path:
        start = predicted_path[0]
        ax.plot(start[0] + 0.5, start[1] + 0.5, 'go', markersize=15, 
               label='Start', zorder=10, markeredgecolor='black', markeredgewidth=2)
    
    goal_pos = None
    if hasattr(env.unwrapped, 'grid'):
        grid = env.unwrapped.grid
        for x in range(width):
            for y in range(height):
                try:
                    cell = grid.get(x, y)
                    if cell and hasattr(cell, 'type') and cell.type == 'goal':
                        goal_pos = (x, y)
                        break
                except:
                    continue
        if goal_pos:
            ax.plot(goal_pos[0] + 0.5, goal_pos[1] + 0.5, 'r*', markersize=20, 
                   label='Goal', zorder=10, markeredgecolor='black', markeredgewidth=1)
    
    margin = 0.5
    ax.set_xlim(-margin, width + margin)
    ax.set_ylim(height + margin, -margin)
    ax.set_aspect('equal')
    ax.set_title(f'Policy Evaluation: {env_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X Position', fontsize=10)
    ax.set_ylabel('Y Position', fontsize=10)
    
    env.close()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    return fig


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate raw DiT policy (no MCTD)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data pickle file')
    parser.add_argument('--env', type=str, required=True,
                       help='Environment name (e.g., MiniGrid-FourRooms-v0)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to evaluate')
    parser.add_argument('--num_denoise_steps', type=int, default=20,
                       help='Number of denoising steps (more = better quality, slower)')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                       help='Guidance scale for denoising')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for visualizations')
    parser.add_argument('--base_seed', type=int, default=42,
                       help='Base seed for environment generation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    model = model.to(args.device)
    
    print(f"Loading test trajectories from {args.data}...")
    with open(args.data, 'rb') as f:
        test_trajectories = pickle.load(f)
    
    print(f"Creating action encoder...")
    model_config_obj = get_model_config()
    action_encoder = ActionEncoder(
        num_actions=model_config_obj.num_actions,
        hidden_dim=model.hidden_dim
    ).to(args.device)
    
    print(f"\n=== Evaluating Raw DiT Policy ===")
    print(f"Denoising steps: {args.num_denoise_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Evaluating on {min(args.num_samples, len(test_trajectories))} samples...\n")
    
    # Results storage
    results = []
    all_actions = []
    train_size = 400  # Assuming 80/20 split
    
    num_samples = min(args.num_samples, len(test_trajectories))
    test_indices = np.random.choice(len(test_trajectories), num_samples, replace=False)
    
    for i, idx in enumerate(test_indices):
        traj = test_trajectories[idx]
        print(f"Sample {i+1}/{num_samples} (trajectory {idx})...")
        
        # Get episode seed
        if 'episode_seed' in traj and traj['episode_seed'] is not None:
            episode_seed = int(traj['episode_seed'])
        else:
            episode_seed = int(args.base_seed + train_size + idx)
        
        # Create environment
        env = gym.make(args.env)
        obs, info = env.reset(seed=episode_seed)
        
        # Prepare state
        full_grid_image = get_full_grid_image(env)
        initial_state = {
            'grid': torch.tensor(full_grid_image, dtype=torch.float32).unsqueeze(0),
            'direction': torch.tensor([obs['direction']], dtype=torch.long),
        }
        initial_state = {k: v.to(args.device) for k, v in initial_state.items()}
        
        # Get optimal path
        optimal_actions = torch.tensor(traj['actions'], dtype=torch.long)
        
        # Run policy
        predicted_actions = direct_policy_decode(
            model, initial_state, device=args.device,
            num_denoise_steps=args.num_denoise_steps,
            guidance_scale=args.guidance_scale
        )
        
        # Execute and evaluate
        positions, success, steps, reward = execute_path(env, predicted_actions, seed=episode_seed)
        
        # Get optimal path positions
        env.reset(seed=episode_seed)
        optimal_positions, optimal_success, optimal_steps, _ = execute_path(env, optimal_actions, seed=episode_seed)
        
        # Collect statistics
        action_counts = Counter(predicted_actions.tolist())
        all_actions.extend(predicted_actions.tolist())
        
        results.append({
            'success': success,
            'steps': steps,
            'reward': reward,
            'optimal_steps': optimal_steps,
            'optimal_success': optimal_success,
            'action_distribution': dict(action_counts),
            'positions': positions,
            'optimal_positions': optimal_positions,
        })
        
        print(f"  Success: {success}, Steps: {steps}, Reward: {reward:.3f}")
        print(f"  Optimal: {optimal_success}, Steps: {optimal_steps}")
        print(f"  Action distribution: {action_counts}")
        
        # Visualize first few samples
        if i < 5:
            fig_path = output_dir / f'policy_sample_{i+1}.png'
            visualize_path(
                args.env, positions, optimal_positions, 
                seed=episode_seed, save_path=str(fig_path)
            )
            print(f"  Saved visualization to {fig_path}")
        
        env.close()
    
    # Compute statistics
    print(f"\n=== Evaluation Statistics ===")
    success_rate = np.mean([r['success'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        avg_successful_steps = np.mean([r['steps'] for r in successful_results])
        avg_optimal_steps = np.mean([r['optimal_steps'] for r in successful_results])
        efficiency = avg_optimal_steps / avg_successful_steps if avg_successful_steps > 0 else 0
    else:
        avg_successful_steps = None
        avg_optimal_steps = None
        efficiency = 0
    
    # Action diversity
    all_action_counts = Counter(all_actions)
    total_actions = len(all_actions)
    action_diversity = {k: (v / total_actions * 100) for k, v in all_action_counts.items()}
    
    print(f"Success Rate: {success_rate:.1%} ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Reward: {avg_reward:.3f}")
    if successful_results:
        print(f"Average Steps (successful): {avg_successful_steps:.1f}")
        print(f"Average Optimal Steps: {avg_optimal_steps:.1f}")
        print(f"Efficiency (optimal/predicted): {efficiency:.2%}")
    
    print(f"\nAction Diversity:")
    for action, pct in sorted(action_diversity.items()):
        action_names = {0: 'turn_left', 1: 'turn_right', 2: 'move_forward', 
                       3: 'pickup', 4: 'drop', 5: 'toggle', 6: 'done'}
        name = action_names.get(action, f'action_{action}')
        print(f"  {action} ({name}): {pct:.1f}%")
    
    # Save summary
    summary_path = output_dir / 'policy_evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=== Raw DiT Policy Evaluation Summary ===\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Environment: {args.env}\n")
        f.write(f"Test Data: {args.data}\n")
        f.write(f"Num Samples: {len(results)}\n")
        f.write(f"Denoising Steps: {args.num_denoise_steps}\n")
        f.write(f"Guidance Scale: {args.guidance_scale}\n\n")
        f.write(f"Success Rate: {success_rate:.1%}\n")
        f.write(f"Average Steps: {avg_steps:.1f}\n")
        f.write(f"Average Reward: {avg_reward:.3f}\n")
        if successful_results:
            f.write(f"Average Steps (successful): {avg_successful_steps:.1f}\n")
            f.write(f"Efficiency: {efficiency:.2%}\n")
        f.write(f"\nAction Diversity:\n")
        for action, pct in sorted(action_diversity.items()):
            action_names = {0: 'turn_left', 1: 'turn_right', 2: 'move_forward', 
                           3: 'pickup', 4: 'drop', 5: 'toggle', 6: 'done'}
            name = action_names.get(action, f'action_{action}')
            f.write(f"  {action} ({name}): {pct:.1f}%\n")
    
    print(f"\nSaved summary to {summary_path}")
    print("\n=== Evaluation Complete ===")


if __name__ == '__main__':
    main()
