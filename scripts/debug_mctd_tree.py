"""
Debug script to understand MCTD tree structure and why paths look identical.

Diagnostics:
- Tree depth and breadth
- Q-value distribution
- Hidden state diversity
- Action diversity across nodes
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import gymnasium as gym
from src.models.diffusion_policy import DiffusionPolicy
from src.models.action_encoder import ActionEncoder
from src.mctd import HiddenSpaceMCTD, MCTDNode
from src.config import get_model_config, get_mctd_config, load_config_from_dict
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


def analyze_tree_structure(node: MCTDNode, depth: int = 0, stats: Dict = None) -> Dict:
    """Analyze tree structure and collect statistics."""
    if stats is None:
        stats = {
            'max_depth': 0,
            'total_nodes': 0,
            'nodes_by_depth': defaultdict(int),
            'q_values': [],
            'noise_levels': [],
            'visits': [],
            'action_sequences': [],  # Store decoded actions from each terminal node
        }
    
    stats['total_nodes'] += 1
    stats['max_depth'] = max(stats['max_depth'], depth)
    stats['nodes_by_depth'][depth] += 1
    
    if node.visits > 0:
        stats['q_values'].append(node.q_value)
        stats['noise_levels'].append(node.noise_level)
        stats['visits'].append(node.visits)
    
    # If terminal, decode to see what action sequence this path leads to
    if node.is_terminal():
        try:
            h_batch = node.hidden_state.unsqueeze(0)  # [1, L, D]
            t_final = torch.tensor([0.0], device=node.hidden_state.device)
            
            with torch.no_grad():
                logits = model.forward(h_batch, node.env_state, t_final)
                actions = logits.argmax(dim=-1)[0].cpu()  # [L]
                actions_tuple = tuple(actions[:10].tolist())  # First 10 actions for comparison
                stats['action_sequences'].append((actions_tuple, node.q_value))
        except Exception as e:
            print(f"  Error decoding node: {e}")
    
    # Recurse on children
    for child in node.children:
        analyze_tree_structure(child, depth + 1, stats)
    
    return stats


def compare_hidden_states(node: MCTDNode, depth: int = 0, max_nodes: int = 20) -> List[float]:
    """Compute distances between hidden states (sample first max_nodes nodes)."""
    distances = []
    hidden_states = []
    node_queue = [(node, 0)]
    visited = 0
    
    while node_queue and visited < max_nodes:
        current, current_depth = node_queue.pop(0)
        hidden_states.append(current.hidden_state.clone().cpu())
        visited += 1
        
        for child in current.children:
            node_queue.append((child, current_depth + 1))
    
    # Compute pairwise distances (just compare root to others)
    if len(hidden_states) > 1:
        root_h = hidden_states[0]
        for i in range(1, len(hidden_states)):
            dist = ((root_h - hidden_states[i]) ** 2).mean().item()
            distances.append(dist)
    
    return distances


def main():
    """Main diagnostic function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug MCTD tree structure')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_epoch20.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/FourRooms_test.pkl',
                       help='Path to test data')
    parser.add_argument('--env', type=str, default='MiniGrid-FourRooms-v0',
                       help='Environment name')
    parser.add_argument('--num_simulations', type=int, default=50,
                       help='Number of MCTD simulations')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--trajectory_idx', type=int, default=0,
                       help='Index of trajectory to test')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}...")
    global model  # For use in analyze_tree_structure
    model = load_model(args.checkpoint, device=args.device)
    
    print(f"Loading test trajectories from {args.data}...")
    with open(args.data, 'rb') as f:
        test_trajectories = pickle.load(f)
    
    print(f"Creating action encoder...")
    model_config_obj = get_model_config()
    action_encoder = ActionEncoder(
        num_actions=model_config_obj.num_actions,
        hidden_dim=model.hidden_dim
    ).to(args.device)
    
    # Select trajectory
    traj = test_trajectories[args.trajectory_idx]
    episode_seed = int(traj.get('episode_seed', 42 + args.trajectory_idx))
    
    print(f"\n=== Testing Trajectory {args.trajectory_idx} (seed={episode_seed}) ===")
    
    # Create environment
    env = gym.make(args.env)
    obs, info = env.reset(seed=episode_seed)
    
    # Prepare initial state
    full_grid_image = get_full_grid_image(env)
    initial_state_mctd = {
        'grid': full_grid_image,
        'direction': obs['direction'],
    }
    
    # Run MCTD search
    print(f"Running MCTD search with {args.num_simulations} simulations...")
    mctd_config = get_mctd_config()
    mctd = HiddenSpaceMCTD(
        policy_model=model,
        env=env,
        action_encoder=action_encoder,
        num_simulations=args.num_simulations,
        guidance_scales=mctd_config.guidance_scales,
        sparse_timesteps=mctd_config.sparse_timesteps,
        denoising_step_size=mctd_config.denoising_step_size,
        reward_alpha=mctd_config.reward_alpha,
        device=args.device,
    )
    
    mctd_actions, search_tree = mctd.search(initial_state_mctd, reference_path=None, use_similarity_reward=False)
    
    print(f"\n=== Tree Structure Analysis ===")
    stats = analyze_tree_structure(search_tree)
    
    print(f"Tree Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Nodes by depth: {dict(stats['nodes_by_depth'])}")
    
    if stats['q_values']:
        q_vals = np.array(stats['q_values'])
        print(f"\nQ-value Statistics:")
        print(f"  Mean: {q_vals.mean():.3f}")
        print(f"  Std: {q_vals.std():.3f}")
        print(f"  Min: {q_vals.min():.3f}")
        print(f"  Max: {q_vals.max():.3f}")
        print(f"  Range: {q_vals.max() - q_vals.min():.3f}")
    
    if stats['noise_levels']:
        noise_levels = np.array(stats['noise_levels'])
        print(f"\nNoise Level Statistics:")
        print(f"  Mean: {noise_levels.mean():.3f}")
        print(f"  Min: {noise_levels.min():.3f}")
        print(f"  Max: {noise_levels.max():.3f}")
    
    print(f"\n=== Action Sequence Diversity ===")
    unique_sequences = {}
    for actions_tuple, q_val in stats['action_sequences']:
        if actions_tuple not in unique_sequences:
            unique_sequences[actions_tuple] = []
        unique_sequences[actions_tuple].append(q_val)
    
    print(f"Total terminal nodes: {len(stats['action_sequences'])}")
    print(f"Unique action sequences (first 10 actions): {len(unique_sequences)}")
    
    if len(unique_sequences) > 0:
        print(f"\nTop action sequences by Q-value:")
        sorted_seqs = sorted(unique_sequences.items(), 
                           key=lambda x: max(x[1]) if x[1] else -float('inf'), 
                           reverse=True)
        for i, (seq, q_vals) in enumerate(sorted_seqs[:5]):
            print(f"  {i+1}. Actions: {seq[:5]}... Q-values: {q_vals[:3]}")
    else:
        print("  ⚠️ No terminal nodes found!")
    
    # Analyze hidden state diversity
    print(f"\n=== Hidden State Diversity ===")
    distances = compare_hidden_states(search_tree, max_nodes=20)
    
    if distances:
        print(f"Mean squared distance from root to other nodes: {np.mean(distances):.6f}")
        print(f"Max distance: {np.max(distances):.6f}")
        print(f"Min distance: {np.min(distances):.6f}")
        print(f"Num nodes compared: {len(distances)}")
    
    # Check root's children diversity
    print(f"\n=== Root Children Analysis ===")
    if search_tree.children:
        print(f"Root has {len(search_tree.children)} children")
        print(f"Root Q-value: {search_tree.q_value:.3f}, visits: {search_tree.visits}")
        
        for i, child in enumerate(search_tree.children):
            print(f"  Child {i}: Q={child.q_value:.3f}, visits={child.visits}, "
                  f"noise={child.noise_level:.3f}, depth={child.depth}")
            
            # Decode child's hidden state to see actions
            try:
                h_batch = child.hidden_state.unsqueeze(0)
                t_test = torch.tensor([child.noise_level], device=args.device)
                logits = model.forward(h_batch, child.env_state, t_test)
                actions = logits.argmax(dim=-1)[0].cpu()
                action_counts = {int(a): (actions == a).sum().item() for a in actions.unique()}
                print(f"    Decoded actions (noise={child.noise_level:.3f}): {action_counts}")
            except Exception as e:
                print(f"    Error decoding: {e}")
    
    # Analyze extract_best_trajectory path
    print(f"\n=== Best Trajectory Path Analysis ===")
    node = search_tree
    path_nodes = [node]
    print(f"Following Q-value path from root...")
    
    while not node.is_terminal() and node.children:
        best_child = max(node.children, key=lambda c: c.q_value)
        print(f"  Depth {node.depth} -> {node.depth + 1}: "
              f"Q={node.q_value:.3f} -> Q={best_child.q_value:.3f}, "
              f"noise={node.noise_level:.3f} -> {best_child.noise_level:.3f}")
        node = best_child
        path_nodes.append(node)
    
    print(f"  Final node: Q={node.q_value:.3f}, noise={node.noise_level:.3f}, terminal={node.is_terminal()}")
    
    # Decode final path
    h_batch = node.hidden_state.unsqueeze(0)
    t_final = torch.tensor([0.0], device=args.device)
    logits = model.forward(h_batch, node.env_state, t_final)
    final_actions = logits.argmax(dim=-1)[0].cpu()
    print(f"  Final decoded actions (first 10): {final_actions[:10].tolist()}")
    action_counts = {int(a): (final_actions == a).sum().item() for a in final_actions.unique()}
    print(f"  Final action distribution: {action_counts}")
    
    env.close()
    print("\n=== Diagnostic Complete ===")


if __name__ == '__main__':
    main()
