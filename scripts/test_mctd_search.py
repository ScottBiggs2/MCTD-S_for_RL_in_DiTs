"""
Test script for MCTD search visualization and comparison.

Shows:
- Speculative paths from search tree
- Chosen path (MCTD best)
- Optimal path (expert)
- Statistics comparison: direct policy vs MCTD
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
from matplotlib.gridspec import GridSpec

# Import minigrid to register environments
try:
    import minigrid
except ImportError:
    print("ERROR: minigrid package not found. Install with: pip install minigrid")
    sys.exit(1)

import gymnasium as gym
from src.models.diffusion_policy import DiffusionPolicy
from src.models.action_encoder import ActionEncoder
from src.mctd import HiddenSpaceMCTD, MCTDNode
from src.environments.trajectory_dataset import TrajectoryDataset
from src.config import get_model_config, get_mctd_config, load_config_from_dict
from src.environments.minigrid_wrapper import get_full_grid_image


def load_model(
    checkpoint_path: str,
    device: str = 'cpu',
    pretrained_state_encoder_path: Optional[str] = None,
    grid_size: int = 19,
) -> DiffusionPolicy:
    """Load trained model from checkpoint.
    
    Attempts to load config from checkpoint. Falls back to default config if not found.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load on
        pretrained_state_encoder_path: Optional path to pretrained state encoder checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load config from checkpoint, otherwise use default
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        # Load config from checkpoint
        experiment_config = load_config_from_dict(checkpoint['config'])
        model_config_dict = experiment_config.to_dict()
        # Extract just model params
        model_config = {
            'num_actions': model_config_dict['num_actions'],
            'hidden_dim': model_config_dict['hidden_dim'],
            'num_layers': model_config_dict['num_layers'],
            'num_heads': model_config_dict['num_heads'],
            'max_seq_len': model_config_dict['max_seq_len'],
            'dropout': model_config_dict['dropout'],
            'grid_size': grid_size,
        }
    else:
        # Use default config (ensure it matches training config)
        model_config_obj = get_model_config()
        model_config = {
            'num_actions': model_config_obj.num_actions,
            'hidden_dim': model_config_obj.hidden_dim,
            'num_layers': model_config_obj.num_layers,
            'num_heads': model_config_obj.num_heads,
            'max_seq_len': model_config_obj.max_seq_len,
            'dropout': model_config_obj.dropout,
            'grid_size': grid_size,
        }
    
    # Create model with config
    model = DiffusionPolicy(**model_config)
    
    # Load weights
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'config' not in checkpoint:  # Don't try to load config dict as weights
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    # Load pretrained state encoder if specified
    if pretrained_state_encoder_path:
        pretrained_path = Path(pretrained_state_encoder_path)
        if not pretrained_path.exists():
            print(f"\n⚠️  Warning: Pretrained state encoder path '{pretrained_state_encoder_path}' not found!")
            print("  Continuing without pretrained encoder...")
        else:
            print(f"\n{'='*60}")
            print(f"Loading pretrained state encoder from {pretrained_state_encoder_path}")
            print(f"{'='*60}")
            state_encoder_checkpoint = torch.load(pretrained_state_encoder_path, map_location=device)
            
            if isinstance(state_encoder_checkpoint, dict) and 'model_state_dict' in state_encoder_checkpoint:
                state_dict = state_encoder_checkpoint['model_state_dict']
                checkpoint_config = state_encoder_checkpoint.get('config', {})
            else:
                state_dict = state_encoder_checkpoint
                checkpoint_config = {}
            
            # Check for hidden_dim mismatch
            if checkpoint_config:
                pretrained_hidden_dim = checkpoint_config.get('hidden_dim')
                current_hidden_dim = model_config.get('hidden_dim')
                if pretrained_hidden_dim and pretrained_hidden_dim != current_hidden_dim:
                    print(f"\n⚠️  WARNING: Hidden dimension mismatch!")
                    print(f"  Pretrained encoder: hidden_dim={pretrained_hidden_dim}")
                    print(f"  Current model: hidden_dim={current_hidden_dim}")
                    print(f"  This may cause loading errors!")
            
            # Load state encoder weights (filter out decoder/auxiliary heads)
            encoder_state_dict = {}
            for k, v in state_dict.items():
                # Skip decoder and auxiliary heads
                if 'decoder' not in k and 'agent_pos_head' not in k and 'goal_pos_head' not in k and 'direction_head' not in k:
                    encoder_state_dict[k] = v
            
            # Load into model's state encoder
            model.state_encoder.load_state_dict(encoder_state_dict, strict=False)
            print("✓ Pretrained state encoder loaded (decoder/auxiliary heads ignored)")
    
    model.eval()
    return model.to(device)


def load_test_trajectories(data_path: str) -> List[Dict]:
    """Load test trajectories from pickle file."""
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


def direct_policy_decode(
    model: DiffusionPolicy,
    state: Dict[str, torch.Tensor],
    device: str = 'cpu',
    num_steps: int = 64,
) -> torch.Tensor:
    """
    Direct policy decode using MDLM iterative unmasking (no hidden-state denoising).
    """
    model.eval()
    with torch.no_grad():
        actions = model.sample(
            state=state,
            seq_len=min(model.max_seq_len, num_steps),
            num_steps=20,
            temperature=1.0,
        )[0]
        actions = torch.clamp(actions, 0, model.num_actions - 1)
        return actions.cpu()


def extract_tree_paths(model: DiffusionPolicy, action_encoder: ActionEncoder,
                      search_tree: MCTDNode, max_paths: int = 5, device: str = 'cpu') -> List[torch.Tensor]:
    """
    Extract multiple speculative paths from search tree.
    
    Returns top paths by Q-value from different branches.
    """
    paths = []
    actions_list = []
    
    def traverse_and_decode(node: MCTDNode, depth: int = 0, max_depth: int = 4):
        if len(actions_list) >= max_paths or depth >= max_depth:
            return
        
        # If terminal or leaf, decode this path
        if node.is_terminal() or (not node.children and node.visits > 0):
            # Decode hidden state to actions
            h_batch = node.hidden_state.unsqueeze(0).to(device)  # [1, L, D]
            t_final = torch.tensor([0.0], device=device)
            
            with torch.no_grad():
                logits = model.forward(h_batch, node.env_state, t_final)  # [1, L, num_actions]
                actions = logits.argmax(dim=-1)[0].cpu()  # [L]
                actions_list.append(actions)
            return
        
        # Sort children by Q-value (highest first)
        if node.children:
            sorted_children = sorted(node.children, key=lambda c: c.q_value if c.visits > 0 else -float('inf'), reverse=True)
            # Take top 2 children per level
            for child in sorted_children[:2]:
                traverse_and_decode(child, depth + 1, max_depth)
    
    traverse_and_decode(search_tree, depth=0, max_depth=4)
    
    # Return unique paths (avoid duplicates)
    seen = set()
    unique_paths = []
    for actions in actions_list:
        # Create a hash of first 10 actions for uniqueness
        path_hash = tuple(actions[:10].tolist())
        if path_hash not in seen:
            seen.add(path_hash)
            unique_paths.append(actions)
            if len(unique_paths) >= max_paths:
                break
    
    return unique_paths


def execute_path(env, actions: torch.Tensor, seed: Optional[int] = None, max_steps: int = 100, debug: bool = False) -> Tuple[List[Tuple[int, int]], bool, int]:
    """
    Execute action sequence in environment and return path positions.
    
    Args:
        env: Gymnasium environment
        actions: Action sequence tensor
        seed: Optional seed for environment reset
        max_steps: Maximum steps to execute
    
    Returns:
        positions: List of (x, y) positions
        success: Whether goal was reached
        steps_taken: Number of steps executed
    """
    # Reset with seed if provided (for reproducibility)
    # Cast to Python int (not numpy int64) for gymnasium compatibility
    if seed is not None:
        obs, info = env.reset(seed=int(seed))
    else:
        obs, info = env.reset()
    
    positions = []
    
    # Get initial position
    if hasattr(env.unwrapped, 'agent_pos'):
        positions.append(tuple(env.unwrapped.agent_pos))
    
    success = False
    steps = 0
    final_reward = 0.0
    
    if debug and len(actions) == 0:
        print(f"      ⚠️ execute_path: Empty actions tensor!")
        return positions, success, steps
    
    if debug:
        print(f"      execute_path: Executing {len(actions)} actions, seed={seed}")
    
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        steps += 1
        final_reward = reward
        
        if hasattr(env.unwrapped, 'agent_pos'):
            pos = env.unwrapped.agent_pos
            positions.append(tuple(pos))
        
        if done:
            # Check success: reward > 0.1 for MiniGrid (as per BFS code)
            if reward > 0.1:
                success = True
            break
        
        if steps >= max_steps:
            break
    
    return positions, success, steps


def visualize_maze_paths(env_name: str, paths: Dict[str, List[Tuple[int, int]]], 
                         seed: Optional[int] = None, save_path: Optional[str] = None):
    """
    Visualize multiple paths on the maze.
    
    Args:
        env_name: Name of MiniGrid environment
        paths: Dict with keys like 'speculative', 'chosen', 'optimal'
               Each value is a list of (x, y) positions
        seed: Optional seed for environment initialization
        save_path: Optional path to save figure
    """
    # Create temporary environment for rendering
    # Cast to Python int (not numpy int64) for gymnasium compatibility
    env = gym.make(env_name)
    if seed is not None:
        obs, info = env.reset(seed=int(seed))
    else:
        obs, info = env.reset()
    
    # Get maze dimensions
    if hasattr(env.unwrapped, 'grid'):
        grid = env.unwrapped.grid
        width = grid.width
        height = grid.height
    else:
        width = height = 8  # Default for Empty-8x8
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw grid with proper cell types
    for x in range(width):
        for y in range(height):
            # Get cell type to color appropriately
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
    
    # Colors for different paths
    colors = {
        'speculative': 'lightblue',
        'chosen': 'green',
        'optimal': 'blue',
        'direct': 'orange',
    }
    
    # Plot paths
    all_positions = []
    for path_name, positions in paths.items():
        if not positions or len(positions) < 2:
            continue
        
        all_positions.extend(positions)
        color = colors.get(path_name, 'black')
        alpha = 0.4 if path_name == 'speculative' else 0.8
        linewidth = 1.5 if path_name == 'speculative' else 2.5
        markersize = 3 if path_name == 'speculative' else 5
        
        # Convert positions to (x, y) coordinates
        xs, ys = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in positions])
        ax.plot(xs, ys, 'o-', color=color, alpha=alpha, linewidth=linewidth,
               label=path_name.capitalize(), markersize=markersize, zorder=5)
    
    # Mark start and goal
    if all_positions:
        start = all_positions[0]
        ax.plot(start[0] + 0.5, start[1] + 0.5, 'go', markersize=15, 
               label='Start', zorder=10, markeredgecolor='black', markeredgewidth=2)
    
    # Try to find goal position from grid
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
    
    # Set proper axis limits
    margin = 0.5
    ax.set_xlim(-margin, width + margin)
    ax.set_ylim(height + margin, -margin)  # Inverted y-axis
    ax.set_aspect('equal')
    ax.set_title(f'MCTD Search Visualization: {env_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X Position', fontsize=10)
    ax.set_ylabel('Y Position', fontsize=10)
    
    env.close()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    return fig


def compare_statistics(results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Compute statistics for comparison.
    
    Args:
        results: Dict with 'direct' and 'mctd' keys, each containing list of result dicts
    
    Returns:
        Statistics dict with metrics for each method
    """
    stats = {}
    
    for method, method_results in results.items():
        success_rates = [r['success'] for r in method_results]
        path_lengths = [r['path_length'] for r in method_results if r['success']]
        
        stats[method] = {
            'success_rate': np.mean(success_rates) if success_rates else 0.0,
            'num_successes': int(np.sum(success_rates)),
            'num_total': len(success_rates),
            'mean_path_length': np.mean(path_lengths) if path_lengths else None,
            'std_path_length': np.std(path_lengths) if path_lengths else None,
        }
    
    return stats


def plot_comparison_statistics(stats: Dict[str, Dict], save_path: Optional[str] = None):
    """Plot comparison statistics chart."""
    methods = list(stats.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Success rate comparison
    ax1 = axes[0]
    success_rates = [stats[m]['success_rate'] for m in methods]
    num_successes = [stats[m]['num_successes'] for m in methods]
    num_total = [stats[m]['num_total'] for m in methods]
    
    bars = ax1.bar(methods, success_rates, color=['skyblue', 'lightcoral'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.set_title('Success Rate Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_xticklabels([m.upper() for m in methods], fontsize=11)
    
    # Add value labels on bars
    for i, (bar, rate, n_succ, n_tot) in enumerate(zip(bars, success_rates, num_successes, num_total)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}\n({n_succ}/{n_tot})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Path length comparison
    ax2 = axes[1]
    mean_lengths = [stats[m]['mean_path_length'] if stats[m]['mean_path_length'] else 0 
                    for m in methods]
    std_lengths = [stats[m]['std_path_length'] if stats[m]['std_path_length'] else 0 
                   for m in methods]
    
    bars = ax2.bar(methods, mean_lengths, yerr=std_lengths, 
                   color=['skyblue', 'lightcoral'], alpha=0.8, capsize=8, 
                   edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2})
    ax2.set_ylabel('Mean Path Length (steps)', fontsize=12, fontweight='bold')
    ax2.set_title('Path Length Comparison (successful only)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_xticklabels([m.upper() for m in methods], fontsize=11)
    
    # Add value labels
    for i, (bar, mean_len, std_len) in enumerate(zip(bars, mean_lengths, std_lengths)):
        height = bar.get_height()
        y_pos = height + std_len + 1 if height > 0 else 0.5
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{mean_len:.1f}±{std_len:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    return fig


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MCTD search visualization')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_epoch0.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/Empty-8x8_test.pkl',
                       help='Path to test data')
    parser.add_argument('--env', type=str, default='MiniGrid-Empty-8x8-v0',
                       help='Environment name')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to visualize')
    parser.add_argument('--num_test', type=int, default=20,
                       help='Number of samples for statistics')
    parser.add_argument('--num_simulations', type=int, default=None,
                       help='Number of MCTD simulations (overrides config default)')
    parser.add_argument('--exploration_const', type=float, default=None,
                       help='UCT exploration constant (overrides config default)')
    parser.add_argument('--guidance_scales', type=float, nargs='+', default=None,
                       help='Guidance scales for expansion, e.g., --guidance_scales 0.0 0.5 1.0')
    parser.add_argument('--denoising_step_size', type=float, default=None,
                       help='Denoising step size (overrides config default)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for figures')
    parser.add_argument('--base_seed', type=int, default=42,
                       help='Base seed for environment generation (should match data generation)')
    parser.add_argument('--use_distance_reward', action='store_true',
                       help='Enable intermediate rewards based on Manhattan distance to goal')
    parser.add_argument('--distance_reward_scale', type=float, default=0.1,
                       help='Scale factor for distance-based rewards (default 0.1)')
    parser.add_argument('--pretrained_state_encoder', type=str, default=None,
                       help='Path to pretrained state encoder checkpoint (from Stage 1 pretraining)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Infer grid size from environment name
    grid_size_map = {
        'MiniGrid-FourRooms-v0': 19,
        'MiniGrid-Empty-8x8-v0': 8,
        'FourRooms': 19,
        'Empty-8x8': 8,
    }
    grid_size = grid_size_map.get(args.env, 19)

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(
        args.checkpoint,
        device=args.device,
        pretrained_state_encoder_path=args.pretrained_state_encoder,
        grid_size=grid_size,
    )
    model = model.to(args.device)
    
    print(f"Loading test trajectories from {args.data}...")
    test_trajectories = load_test_trajectories(args.data)
    
    print(f"Creating action encoder...")
    # Action encoder must match model's hidden_dim and num_actions
    # Get from model (which uses config) to ensure consistency
    model_config_obj = get_model_config()
    action_encoder = ActionEncoder(
        num_actions=model_config_obj.num_actions,
        hidden_dim=model.hidden_dim  # Use model's actual hidden_dim
    ).to(args.device)
    print(f"Creating action encoder with num_actions={model_config_obj.num_actions}, hidden_dim={model.hidden_dim} (matching model)...")
    
    print(f"\n=== Running MCTD Search Tests ===")
    
    # Results storage
    all_results = {'direct': [], 'mctd': []}
    
    # Test on multiple samples for statistics
    num_test = min(args.num_test, len(test_trajectories))
    test_indices = np.random.choice(len(test_trajectories), num_test, replace=False)
    
    # Base seed from data generation (reconstruct seed for each trajectory)
    # Trajectories were generated with: seed = base_seed + episode_idx
    # Test set starts from train_split, so we need: seed = base_seed + (episode_idx + train_size)
    # Since we don't know train_size, we'll use: seed = base_seed + (test_index + offset)
    # For now, use trajectory index as proxy (may need adjustment based on actual split)
    train_size = 400  # Assuming 80/20 split for 500 episodes
    
    for i, idx in enumerate(test_indices):
        traj = test_trajectories[idx]
        print(f"\nSample {i+1}/{num_test} (trajectory {idx})...")
        
        # Use stored seed from trajectory if available (after fixing data generation)
        # Otherwise, reconstruct seed as fallback
        if 'episode_seed' in traj and traj['episode_seed'] is not None:
            episode_seed = int(traj['episode_seed'])  # Use stored seed from trajectory
        else:
            # Fallback: reconstruct seed (won't match exactly due to shuffling)
            # Trajectories are shuffled after generation, so we can't reliably
            # reconstruct the original episode seed without the stored seed.
            episode_seed = int(args.base_seed + train_size + idx)  # Cast to Python int
            print(f"  Warning: No stored seed in trajectory, using reconstructed seed {episode_seed}")
        
        # Create environment with specific seed
        env = gym.make(args.env)
        obs, info = env.reset(seed=episode_seed)
        
        # Use full grid image for initial state (matching training data format)
        # This is critical: if model was trained on full grid, we must use full grid here
        full_grid_image = get_full_grid_image(env)  # (H, W, 3) e.g., (19, 19, 3) for FourRooms
        
        initial_state_mctd = {
            'grid': full_grid_image,  # Full grid image (H, W, 3) numpy array
            'direction': obs['direction'],  # int
        }
        
        # Convert to tensor format for direct policy
        initial_state = {
            'grid': torch.tensor(initial_state_mctd['grid'], dtype=torch.float32).unsqueeze(0),
            'direction': torch.tensor([initial_state_mctd['direction']], dtype=torch.long),
        }
        initial_state = {k: v.to(args.device) for k, v in initial_state.items()}
        
        # Get optimal path (expert) from trajectory
        optimal_actions = torch.tensor(traj['actions'], dtype=torch.long)
        
        # 1. Direct policy
        print("  Running direct policy...")
        direct_actions = direct_policy_decode(model, initial_state, device=args.device)
        
        # Debug: Check action predictions
        print(f"    Direct policy: {len(direct_actions)} actions predicted")
        if len(direct_actions) > 0:
            action_counts = {a.item(): (direct_actions == a).sum().item() for a in direct_actions.unique()}
            print(f"    Action distribution: {action_counts}")
            print(f"    First 10 actions: {direct_actions[:10].tolist()}")
            invalid_actions = (direct_actions < 0) | (direct_actions >= 7)
            if invalid_actions.any():
                print(f"    ⚠️ WARNING: {invalid_actions.sum().item()} invalid actions detected!")
        else:
            print(f"    ⚠️ WARNING: No actions predicted!")
        
        direct_positions, direct_success, direct_steps = execute_path(env, direct_actions, seed=episode_seed)
        print(f"    Direct policy result: success={direct_success}, steps={direct_steps}, path_length={direct_steps}")
        
        all_results['direct'].append({
            'success': direct_success,
            'path_length': len(direct_positions) - 1 if direct_positions else 0,
            'actions': direct_actions,
            'positions': direct_positions,
        })
        
        # 2. MCTD search
        print("  Running MCTD search...")
        # Load MCTD config from central config module
        # Command-line args override defaults
        mctd_config = get_mctd_config()
        
        # Override config with command-line arguments if provided
        num_simulations = args.num_simulations if args.num_simulations is not None else mctd_config.num_simulations
        exploration_const = args.exploration_const if args.exploration_const is not None else mctd_config.exploration_const
        guidance_scales = args.guidance_scales if args.guidance_scales is not None else mctd_config.guidance_scales
        denoising_step_size = args.denoising_step_size if args.denoising_step_size is not None else mctd_config.denoising_step_size
        
        # Get distance reward settings
        use_distance_reward = args.use_distance_reward if hasattr(args, 'use_distance_reward') else False
        distance_reward_scale = args.distance_reward_scale if hasattr(args, 'distance_reward_scale') else 0.1
        
        print(f"    MCTD parameters: sims={num_simulations}, exploration={exploration_const:.2f}, "
              f"guidance={guidance_scales}, step_size={denoising_step_size:.2f}")
        if use_distance_reward:
            print(f"    Distance rewards: ENABLED (scale={distance_reward_scale:.2f})")
        else:
            print(f"    Distance rewards: DISABLED")
        
        mctd = HiddenSpaceMCTD(
            policy_model=model,
            env=env,
            action_encoder=action_encoder,
            num_simulations=num_simulations,
            exploration_const=exploration_const,
            guidance_scales=guidance_scales,
            sparse_timesteps=mctd_config.sparse_timesteps,  # From config
            denoising_step_size=denoising_step_size,
            reward_alpha=mctd_config.reward_alpha,  # From config
            device=args.device,
            initial_seed=episode_seed,  # CRITICAL: Pass seed for consistent resets
        )
        
        # Disable cosine similarity reward (user requested removal)
        # Distance rewards are already set above in the print statement
        mctd_actions, search_tree = mctd.search(
            initial_state_mctd, 
            reference_path=None, 
            use_similarity_reward=False,
            use_distance_reward=use_distance_reward,
            distance_reward_scale=distance_reward_scale
        )
        
        # Debug: Check MCTD action predictions
        if mctd_actions is not None:
            mctd_actions_tensor = torch.tensor(mctd_actions) if not isinstance(mctd_actions, torch.Tensor) else mctd_actions
            print(f"    MCTD search: {len(mctd_actions_tensor)} actions predicted")
            if len(mctd_actions_tensor) > 0:
                action_counts = {a.item(): (mctd_actions_tensor == a).sum().item() for a in mctd_actions_tensor.unique()}
                print(f"    Action distribution: {action_counts}")
                print(f"    First 10 actions: {mctd_actions_tensor[:10].tolist()}")
                invalid_actions = (mctd_actions_tensor < 0) | (mctd_actions_tensor >= 7)
                if invalid_actions.any():
                    print(f"    ⚠️ WARNING: {invalid_actions.sum().item()} invalid actions detected!")
                
                # Check search tree statistics
                if search_tree is not None:
                    print(f"    Search tree: root visits={search_tree.visits}, Q-value={search_tree.q_value:.3f}")
                    if search_tree.children:
                        print(f"    Best child Q-value: {max(c.q_value for c in search_tree.children if c.visits > 0):.3f}")
            else:
                print(f"    ⚠️ WARNING: MCTD search returned empty actions!")
        else:
            print(f"    ⚠️ WARNING: MCTD search returned None!")
            mctd_actions_tensor = torch.tensor([], dtype=torch.long)
        
        mctd_positions, mctd_success, mctd_steps = execute_path(env, mctd_actions, seed=episode_seed)
        print(f"    MCTD result: success={mctd_success}, steps={mctd_steps}, path_length={mctd_steps}")
        
        all_results['mctd'].append({
            'success': mctd_success,
            'path_length': len(mctd_positions) - 1 if mctd_positions else 0,
            'actions': mctd_actions,
            'positions': mctd_positions,
            'search_tree': search_tree,
        })
        
        # Visualize first few samples
        if i < args.num_samples:
            print(f"  Visualizing sample {i+1}...")
            
            # Extract multiple speculative paths from tree
            speculative_actions = extract_tree_paths(model, action_encoder, search_tree, 
                                                    max_paths=3, device=args.device)
            
            # Get optimal and direct paths (reset env with seed)
            env.reset(seed=episode_seed)
            optimal_positions, _, _ = execute_path(env, optimal_actions, seed=episode_seed)
            
            env.reset(seed=episode_seed)
            direct_positions_vis, _, _ = execute_path(env, direct_actions, seed=episode_seed)
            
            # Execute speculative paths
            speculative_positions_list = []
            for spec_actions in speculative_actions[:2]:  # Show up to 2 speculative paths
                env.reset(seed=episode_seed)
                spec_pos, _, _ = execute_path(env, spec_actions, seed=episode_seed)
                if spec_pos:
                    speculative_positions_list.append(spec_pos)
            
            # Combine all paths for visualization
            paths = {
                'chosen': mctd_positions if mctd_positions and len(mctd_positions) >= 2 else None,  # MCTD best path
                'optimal': optimal_positions,  # Expert optimal path
                'direct': direct_positions_vis,  # Direct policy path
            }
            
            # Add first speculative path if available
            if speculative_positions_list:
                paths['speculative'] = speculative_positions_list[0]
            
            # Remove None/invalid paths to avoid visualization errors
            paths = {k: v for k, v in paths.items() if v is not None and len(v) >= 2}
            
            fig_path = output_dir / f'sample_{i+1}_paths.png'
            visualize_maze_paths(args.env, paths, seed=episode_seed, save_path=str(fig_path))
            print(f"    Saved visualization to {fig_path}")
        
        env.close()
    
    # Compute and plot statistics
    print(f"\n=== Computing Statistics ===")
    stats = compare_statistics(all_results)
    
    print("\nStatistics:")
    for method in ['direct', 'mctd']:
        s = stats[method]
        print(f"  {method.upper()}:")
        print(f"    Success Rate: {s['success_rate']:.2%} ({s['num_successes']}/{s['num_total']})")
        if s['mean_path_length']:
            print(f"    Mean Path Length: {s['mean_path_length']:.1f} ± {s['std_path_length']:.1f}")
    
    # Plot comparison chart
    fig_stats = output_dir / 'comparison_statistics.png'
    plot_comparison_statistics(stats, save_path=str(fig_stats))
    print(f"\nSaved statistics chart to {fig_stats}")
    
    print("\n=== Test Complete ===")


if __name__ == '__main__':
    main()
