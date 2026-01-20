"""
Simple script to check what actions the model predicts vs expert.
No environment interaction to avoid segfaults.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import pickle
from collections import Counter

from src.models.diffusion_policy import DiffusionPolicy
from src.config import load_config_from_dict, get_model_config


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load trained model from checkpoint.
    
    Uses config from checkpoint if available, otherwise uses default config.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load config from checkpoint
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
        # Use default config
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
    
    # Load weights
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'config' not in checkpoint:
            model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model.to(device)


def main():
    checkpoint = 'checkpoints/best_model_epoch48.pt'
    data_path = 'data/FourRooms_test.pkl'
    device = 'cpu'  # Use CPU to avoid MPS issues
    
    print("="*60)
    print("ACTION PREDICTION ANALYSIS")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from {checkpoint}...")
    model = load_model(checkpoint, device=device)
    
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Analyze predictions
    model.eval()
    action_counts = Counter()
    expert_counts = Counter()
    match_counts = Counter()
    
    print("\n" + "="*60)
    print("Comparing Model Predictions vs Expert")
    print("="*60)
    
    for i in range(min(10, len(trajectories))):
        traj = trajectories[i]
        expert_actions = np.array(traj['actions'])
        
        # Get first state from trajectory
        if len(traj['states']) > 0:
            first_state = traj['states'][0]
            grid = torch.tensor(first_state['grid'], dtype=torch.float32).reshape(7, 7, 3).unsqueeze(0)
            direction = torch.tensor([first_state['direction']], dtype=torch.long)
        else:
            # Fallback: use zeros
            grid = torch.zeros(1, 7, 7, 3)
            direction = torch.tensor([0])
        
        state = {
            'grid': grid,
            'direction': direction,
        }
        
        # Get model prediction
        with torch.no_grad():
            seq_len = model.num_tokens
            
            # Initialize
            init_actions = torch.full((1, seq_len), 2, dtype=torch.long, device=device)
            init_hidden = model.action_encoder(init_actions)
            noise = torch.randn_like(init_hidden)
            hidden_state = init_hidden + 0.5 * noise
            
            # Denoise
            num_steps = 20
            for step in range(num_steps):
                t = 1.0 - (step + 1) / num_steps
                t_tensor = torch.tensor([t], device=device)
                hidden_state = model.denoise_step(hidden_state, state, t_tensor, guidance_scale=1.0)
            
            # Decode
            t_final = torch.tensor([0.0], device=device)
            logits = model.forward(hidden_state, state, t_final)
            pred_actions = logits.argmax(dim=-1)[0].cpu().numpy()
        
        # Compare first 15 actions
        compare_len = min(15, len(expert_actions), len(pred_actions))
        expert_subset = expert_actions[:compare_len]
        pred_subset = pred_actions[:compare_len]
        
        # Count actions
        for a in pred_subset:
            action_counts[int(a)] += 1
        for a in expert_subset:
            expert_counts[int(a)] += 1
        
        # Count matches
        matches = sum(1 for a, b in zip(expert_subset, pred_subset) if a == b)
        match_counts[matches] += 1
        
        print(f"\nSample {i+1}:")
        print(f"  Expert: {expert_subset.tolist()}")
        print(f"  Model:  {pred_subset.tolist()}")
        print(f"  Matches: {matches}/{compare_len} ({100*matches/compare_len:.1f}%)")
        
        # Check for patterns
        if compare_len > 5:
            # Check if all same
            if len(set(pred_subset)) == 1:
                print(f"  ⚠️  Model predicts same action ({pred_subset[0]}) for all positions!")
            # Check if mostly one action
            most_common = Counter(pred_subset).most_common(1)[0]
            if most_common[1] > compare_len * 0.7:
                print(f"  ⚠️  Model heavily favors action {most_common[0]} ({100*most_common[1]/compare_len:.1f}%)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nModel Action Distribution:")
    total_model = sum(action_counts.values())
    for action in sorted(action_counts.keys()):
        pct = 100 * action_counts[action] / total_model if total_model > 0 else 0
        print(f"  Action {action}: {action_counts[action]} ({pct:.1f}%)")
    
    print("\nExpert Action Distribution:")
    total_expert = sum(expert_counts.values())
    for action in sorted(expert_counts.keys()):
        pct = 100 * expert_counts[action] / total_expert if total_expert > 0 else 0
        print(f"  Action {action}: {expert_counts[action]} ({pct:.1f}%)")
    
    print("\nMatch Statistics:")
    total_samples = sum(match_counts.values())
    avg_matches = sum(k * v for k, v in match_counts.items()) / total_samples if total_samples > 0 else 0
    print(f"  Average matches per sample: {avg_matches:.1f}/15")
    
    # Check for inversion patterns
    print("\n" + "="*60)
    print("INVERSION CHECK")
    print("="*60)
    print("If actions are inverted, we'd see high match rates with swapped actions.")
    print("Current match rate suggests:", end=" ")
    if avg_matches < 2:
        print("❌ Very poor alignment (likely weak model or bug)")
    elif avg_matches < 5:
        print("⚠️  Poor alignment (weak model)")
    else:
        print("✅ Reasonable alignment")


if __name__ == '__main__':
    main()
