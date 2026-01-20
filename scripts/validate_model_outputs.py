"""
Validate model outputs to check for action inversion or other bugs.

Tests:
1. Are actions being inverted (0↔1, etc.)?
2. What actions does the model predict vs expert?
3. Are predictions reasonable or completely random?
4. Check action space mapping
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pickle
from collections import Counter
import gymnasium as gym

# Import minigrid
try:
    import minigrid
except ImportError:
    print("ERROR: minigrid package not found")
    sys.exit(1)

from src.models.diffusion_policy import DiffusionPolicy
from src.models.action_encoder import ActionEncoder


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model."""
    config = {
        'num_actions': 7,
        'hidden_dim': 128,
        'num_layers': 6,
        'num_heads': 4,
        'num_tokens': 49,
        'max_seq_len': 64,
        'dropout': 0.1,
    }
    
    model = DiffusionPolicy(**config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model.to(device)


def test_action_space():
    """Test MiniGrid action space to understand mapping."""
    print("="*60)
    print("TEST 1: MiniGrid Action Space")
    print("="*60)
    
    env = gym.make('MiniGrid-FourRooms-v0')
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")
    
    # Test each action
    obs, info = env.reset(seed=42)
    initial_pos = env.unwrapped.agent_pos
    initial_dir = env.unwrapped.agent_dir
    
    print(f"\nInitial position: {initial_pos}, direction: {initial_dir}")
    print("\nTesting actions:")
    
    for action in range(env.action_space.n):
        test_env = gym.make('MiniGrid-FourRooms-v0')
        test_obs, _ = test_env.reset(seed=42)
        test_obs, reward, terminated, truncated, info = test_env.step(action)
        new_pos = test_env.unwrapped.agent_pos
        new_dir = test_env.unwrapped.agent_dir
        
        pos_changed = (new_pos != initial_pos).any()
        dir_changed = new_dir != initial_dir
        
        print(f"  Action {action}: pos={new_pos} (changed={pos_changed}), dir={new_dir} (changed={dir_changed})")
        test_env.close()
    
    env.close()
    print("\nMiniGrid action mapping (standard):")
    print("  0 = turn left")
    print("  1 = turn right")
    print("  2 = move forward")
    print("  3-6 = other actions (pickup, drop, toggle, done)")


def test_model_predictions(model, trajectories, device='cpu', num_samples=5):
    """Test what actions the model predicts vs expert."""
    print("\n" + "="*60)
    print("TEST 2: Model Predictions vs Expert")
    print("="*60)
    
    model.eval()
    
    action_counts = Counter()
    expert_action_counts = Counter()
    
    for i in range(min(num_samples, len(trajectories))):
        traj = trajectories[i]
        expert_actions = traj['actions']
        
        # Get episode seed
        if 'episode_seed' in traj:
            episode_seed = int(traj['episode_seed'])
        else:
            episode_seed = 42 + i
        
        # Create environment and get initial state
        env = gym.make('MiniGrid-FourRooms-v0')
        obs, info = env.reset(seed=episode_seed)
        
        initial_state = {
            'grid': torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0),
            'direction': torch.tensor([obs['direction']], dtype=torch.long),
        }
        initial_state = {k: v.to(device) for k, v in initial_state.items()}
        
        # Get model prediction
        with torch.no_grad():
            seq_len = model.num_tokens
            
            # Initialize with forward actions
            init_actions = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
            init_hidden = model.action_encoder(init_actions)
            noise = torch.randn_like(init_hidden)
            hidden_state = init_hidden + 0.5 * noise
            
            # Denoise
            num_steps = 20
            for step in range(num_steps):
                t = 1.0 - (step + 1) / num_steps
                t_tensor = torch.tensor([t], device=device)
                hidden_state = model.denoise_step(hidden_state, initial_state, t_tensor, guidance_scale=1.0)
            
            # Decode
            t_final = torch.tensor([0.0], device=device)
            logits = model.forward(hidden_state, initial_state, t_final)
            pred_actions = logits.argmax(dim=-1)[0].cpu().numpy()
        
        # Count actions
        for action in pred_actions[:len(expert_actions)]:
            action_counts[int(action)] += 1
        
        for action in expert_actions:
            expert_action_counts[int(action)] += 1
        
        # Compare first 10 actions
        print(f"\nSample {i+1}:")
        print(f"  Expert (first 10): {expert_actions[:10]}")
        print(f"  Model (first 10):  {pred_actions[:10].tolist()}")
        
        # Check for inversion patterns
        if len(expert_actions) > 0 and len(pred_actions) > 0:
            # Check if actions are swapped (0↔1, etc.)
            matches = sum(1 for a, b in zip(expert_actions[:10], pred_actions[:10]) if a == b)
            print(f"  Exact matches (first 10): {matches}/10")
            
            # Check if all same action
            unique_pred = len(set(pred_actions[:10]))
            unique_expert = len(set(expert_actions[:10]))
            print(f"  Unique actions: model={unique_pred}, expert={unique_expert}")
        
        env.close()
    
    print(f"\n{'='*60}")
    print("Action Distribution Summary")
    print(f"{'='*60}")
    print("Model predictions:")
    for action in sorted(action_counts.keys()):
        print(f"  Action {action}: {action_counts[action]} times ({100*action_counts[action]/sum(action_counts.values()):.1f}%)")
    
    print("\nExpert trajectories:")
    for action in sorted(expert_action_counts.keys()):
        print(f"  Action {action}: {expert_action_counts[action]} times ({100*expert_action_counts[action]/sum(expert_action_counts.values()):.1f}%)")


def test_action_inversion(model, trajectories, device='cpu', num_samples=3):
    """Test if actions are systematically inverted."""
    print("\n" + "="*60)
    print("TEST 3: Action Inversion Check")
    print("="*60)
    
    # Test common inversion patterns
    inversions = {
        '0↔1': lambda a: 1 if a == 0 else (0 if a == 1 else a),
        '0↔2': lambda a: 2 if a == 0 else (0 if a == 2 else a),
        '1↔2': lambda a: 2 if a == 1 else (1 if a == 2 else a),
        'all_rotated': lambda a: (a + 1) % 3 if a < 3 else a,
    }
    
    for inv_name, inv_func in inversions.items():
        matches = 0
        total = 0
        
        for i in range(min(num_samples, len(trajectories))):
            traj = trajectories[i]
            expert_actions = traj['actions']
            
            if 'episode_seed' in traj:
                episode_seed = int(traj['episode_seed'])
            else:
                episode_seed = 42 + i
            
            env = gym.make('MiniGrid-FourRooms-v0')
            obs, info = env.reset(seed=episode_seed)
            
            initial_state = {
                'grid': torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0),
                'direction': torch.tensor([obs['direction']], dtype=torch.long),
            }
            initial_state = {k: v.to(device) for k, v in initial_state.items()}
            
            with torch.no_grad():
                seq_len = model.num_tokens
                init_actions = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
                init_hidden = model.action_encoder(init_actions)
                noise = torch.randn_like(init_hidden)
                hidden_state = init_hidden + 0.5 * noise
                
                num_steps = 20
                for step in range(num_steps):
                    t = 1.0 - (step + 1) / num_steps
                    t_tensor = torch.tensor([t], device=device)
                    hidden_state = model.denoise_step(hidden_state, initial_state, t_tensor, guidance_scale=1.0)
                
                t_final = torch.tensor([0.0], device=device)
                logits = model.forward(hidden_state, initial_state, t_final)
                pred_actions = logits.argmax(dim=-1)[0].cpu().numpy()
            
            # Check if inverted actions match expert
            for pred, expert in zip(pred_actions[:len(expert_actions)], expert_actions):
                inverted = inv_func(int(pred))
                if inverted == expert:
                    matches += 1
                total += 1
            
            env.close()
        
        match_rate = matches / total if total > 0 else 0
        print(f"{inv_name:15s}: {matches}/{total} matches ({100*match_rate:.1f}%)")
        if match_rate > 0.3:  # If >30% match, might be inverted
            print(f"  ⚠️  Possible inversion pattern detected!")


def test_model_confidence(model, trajectories, device='cpu', num_samples=5):
    """Check model confidence in predictions."""
    print("\n" + "="*60)
    print("TEST 4: Model Confidence Analysis")
    print("="*60)
    
    model.eval()
    
    all_entropies = []
    all_max_probs = []
    
    for i in range(min(num_samples, len(trajectories))):
        traj = trajectories[i]
        
        if 'episode_seed' in traj:
            episode_seed = int(traj['episode_seed'])
        else:
            episode_seed = 42 + i
        
        env = gym.make('MiniGrid-FourRooms-v0')
        obs, info = env.reset(seed=episode_seed)
        
        initial_state = {
            'grid': torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0),
            'direction': torch.tensor([obs['direction']], dtype=torch.long),
        }
        initial_state = {k: v.to(device) for k, v in initial_state.items()}
        
        with torch.no_grad():
            seq_len = model.num_tokens
            init_actions = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
            init_hidden = model.action_encoder(init_actions)
            noise = torch.randn_like(init_hidden)
            hidden_state = init_hidden + 0.5 * noise
            
            num_steps = 20
            for step in range(num_steps):
                t = 1.0 - (step + 1) / num_steps
                t_tensor = torch.tensor([t], device=device)
                hidden_state = model.denoise_step(hidden_state, initial_state, t_tensor, guidance_scale=1.0)
            
            t_final = torch.tensor([0.0], device=device)
            logits = model.forward(hidden_state, initial_state, t_final)
            
            # Compute probabilities
            probs = torch.softmax(logits, dim=-1)[0]  # [seq_len, num_actions]
            
            # Compute entropy (higher = less confident)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            all_entropies.extend(entropy.cpu().numpy().tolist())
            
            # Max probability (higher = more confident)
            max_probs = probs.max(dim=-1)[0]
            all_max_probs.extend(max_probs.cpu().numpy().tolist())
        
        env.close()
    
    print(f"Entropy stats: mean={np.mean(all_entropies):.3f}, std={np.std(all_entropies):.3f}")
    print(f"  (Higher entropy = less confident, max={np.log(7):.3f} for uniform)")
    print(f"Max prob stats: mean={np.mean(all_max_probs):.3f}, std={np.std(all_max_probs):.3f}")
    print(f"  (Higher max prob = more confident, 1.0 = 100% certain)")
    
    if np.mean(all_entropies) > np.log(7) * 0.9:
        print("  ⚠️  Model is very uncertain (near-uniform predictions)")
    elif np.mean(all_max_probs) < 0.3:
        print("  ⚠️  Model has low confidence in predictions")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_epoch48.pt')
    parser.add_argument('--data', type=str, default='data/FourRooms_test.pkl')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--num_samples', type=int, default=5)
    
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL OUTPUT VALIDATION")
    print("="*60)
    
    # Test action space
    test_action_space()
    
    # Load model and data
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, device=args.device)
    
    print(f"Loading test data from {args.data}...")
    with open(args.data, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Run tests
    test_model_predictions(model, trajectories, device=args.device, num_samples=args.num_samples)
    test_action_inversion(model, trajectories, device=args.device, num_samples=args.num_samples)
    test_model_confidence(model, trajectories, device=args.device, num_samples=args.num_samples)
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
