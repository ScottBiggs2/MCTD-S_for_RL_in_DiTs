"""
Debug script to understand why model inference is failing.

Checks:
1. What actions does the model generate?
2. Are actions valid (0-6)?
3. Do paths execute successfully?
4. What does the model output look like?
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pickle
from pathlib import Path

# Import minigrid
try:
    import minigrid
except ImportError:
    print("ERROR: minigrid package not found")
    sys.exit(1)

import gymnasium as gym
from src.models.diffusion_policy import DiffusionPolicy
from src.models.action_encoder import ActionEncoder


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model."""
    config = {
        'num_actions': 7,
        'hidden_dim': 128,
        'num_layers': 8,
        'num_heads': 4,
        'num_tokens': 49,
        'max_seq_len': 64,
        'dropout': 0.1,
    }
    
    model = DiffusionPolicy(**config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model.to(device)


def test_direct_decode(model, state, device='cpu', num_steps=64):
    """Test direct policy decode with debugging."""
    model.eval()
    with torch.no_grad():
        # Start with noisy hidden state
        seq_len = model.num_tokens
        hidden_state = torch.randn(1, seq_len, model.hidden_dim, device=device)
        
        print(f"Initial hidden state shape: {hidden_state.shape}")
        print(f"Initial hidden state stats: mean={hidden_state.mean():.4f}, std={hidden_state.std():.4f}")
        
        # Denoise in steps
        num_denoise_steps = 10
        for i in range(num_denoise_steps):
            t = 1.0 - (i + 1) / num_denoise_steps
            t_tensor = torch.tensor([t], device=device)
            
            hidden_state = model.denoise_step(hidden_state, state, t_tensor, guidance_scale=1.0)
            
            if i == 0 or i == num_denoise_steps - 1:
                print(f"Step {i+1}/{num_denoise_steps}, t={t:.2f}: hidden_state mean={hidden_state.mean():.4f}, std={hidden_state.std():.4f}")
        
        # Final decode to actions
        t_final = torch.tensor([0.0], device=device)
        logits = model.forward(hidden_state, state, t_final)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits stats: mean={logits.mean():.4f}, std={logits.std():.4f}")
        print(f"Logits range: min={logits.min():.4f}, max={logits.max():.4f}")
        
        # Check action distribution
        probs = torch.softmax(logits, dim=-1)
        print(f"Action probabilities (first 10 tokens):")
        for i in range(min(10, logits.shape[1])):
            top_actions = probs[0, i].topk(3)
            print(f"  Token {i}: top actions = {top_actions.indices.tolist()} with probs {top_actions.values.tolist()}")
        
        actions = logits.argmax(dim=-1)[0]  # [seq_len]
        print(f"Generated actions shape: {actions.shape}")
        print(f"Action values: min={actions.min()}, max={actions.max()}, unique={len(actions.unique())}")
        print(f"First 20 actions: {actions[:20].tolist()}")
        
        # Check if all actions are valid (0-6)
        valid = (actions >= 0) & (actions < 7)
        print(f"Valid actions: {valid.sum()}/{len(actions)} ({100*valid.float().mean():.1f}%)")
        
        if not valid.all():
            invalid_indices = (~valid).nonzero(as_tuple=True)[0]
            print(f"  Invalid action indices: {invalid_indices[:10].tolist()}")
            print(f"  Invalid action values: {actions[invalid_indices[:10]].tolist()}")
        
        return actions.cpu()


def test_execution(env, actions, seed=None):
    """Test action execution with detailed output."""
    if seed is not None:
        obs, info = env.reset(seed=int(seed))
    else:
        obs, info = env.reset()
    
    positions = []
    rewards = []
    actions_executed = []
    
    print(f"Starting position: {env.unwrapped.agent_pos}")
    
    for i, action in enumerate(actions):
        action_int = int(action)
        
        if action_int < 0 or action_int >= env.action_space.n:
            print(f"  Step {i}: INVALID ACTION {action_int} (skipping)")
            continue
        
        obs, reward, terminated, truncated, info = env.step(action_int)
        done = terminated or truncated
        
        positions.append(tuple(env.unwrapped.agent_pos))
        rewards.append(reward)
        actions_executed.append(action_int)
        
        if done:
            print(f"  Step {i}: action={action_int}, reward={reward:.3f}, done=True, pos={env.unwrapped.agent_pos}")
            break
        
        if i >= 100:  # Safety limit
            break
    
    print(f"Executed {len(actions_executed)} actions")
    print(f"Final position: {env.unwrapped.agent_pos}")
    print(f"Total reward: {sum(rewards):.3f}")
    print(f"Final reward: {rewards[-1] if rewards else 0:.3f}")
    
    # Check success
    success = False
    if rewards:
        final_reward = rewards[-1]
        success = final_reward > 0.1
        print(f"Success check: final_reward={final_reward:.3f} > 0.1? {success}")
    
    return positions, success, len(actions_executed)


def main():
    """Main debug function."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_epoch37.pt')
    parser.add_argument('--data', type=str, default='data/FourRooms_test.pkl')
    parser.add_argument('--env', type=str, default='MiniGrid-FourRooms-v0')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--num_samples', type=int, default=3)
    
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL INFERENCE DEBUG")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    
    # Load test data
    print(f"Loading test data from {args.data}...")
    with open(args.data, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Test on a few samples
    for i in range(min(args.num_samples, len(trajectories))):
        traj = trajectories[i]
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}/{args.num_samples}")
        print(f"{'='*60}")
        
        # Get trajectory info
        optimal_actions = torch.tensor(traj['actions'], dtype=torch.long)
        print(f"Optimal path length: {len(optimal_actions)}")
        print(f"Optimal actions (first 10): {optimal_actions[:10].tolist()}")
        
        if 'episode_seed' in traj:
            episode_seed = int(traj['episode_seed'])
            print(f"Trajectory seed: {episode_seed}")
        else:
            episode_seed = 42 + 400 + i  # Fallback
            print(f"Using fallback seed: {episode_seed}")
        
        # Create environment
        env = gym.make(args.env)
        obs, info = env.reset(seed=episode_seed)
        
        # Prepare state
        initial_state = {
            'grid': torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0),
            'direction': torch.tensor([obs['direction']], dtype=torch.long),
        }
        initial_state = {k: v.to(args.device) for k, v in initial_state.items()}
        
        print(f"Initial agent position: {env.unwrapped.agent_pos}")
        print(f"Initial agent direction: {obs['direction']}")
        
        # Test optimal path first
        print(f"\n--- Testing OPTIMAL path ---")
        env.reset(seed=episode_seed)
        optimal_positions, optimal_success, optimal_steps = test_execution(env, optimal_actions, seed=episode_seed)
        
        # Test direct policy
        print(f"\n--- Testing DIRECT POLICY ---")
        direct_actions = test_direct_decode(model, initial_state, device=args.device)
        
        env.reset(seed=episode_seed)
        direct_positions, direct_success, direct_steps = test_execution(env, direct_actions, seed=episode_seed)
        
        print(f"\n--- SUMMARY ---")
        print(f"Optimal: success={optimal_success}, steps={optimal_steps}, path_len={len(optimal_positions)}")
        print(f"Direct:  success={direct_success}, steps={direct_steps}, path_len={len(direct_positions)}")
        
        env.close()


if __name__ == '__main__':
    main()
