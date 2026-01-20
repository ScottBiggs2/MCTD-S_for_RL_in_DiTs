"""
Debug script to identify training vs inference mismatch.

This script checks:
1. State format consistency between training and inference
2. Action decoding correctness
3. Model predictions vs ground truth on validation set
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pickle
import numpy as np
from pathlib import Path

from src.models.diffusion_policy import DiffusionPolicy

def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model with correct parameters (matching DiffusionPolicy.__init__ signature)
    model = DiffusionPolicy(
        num_actions=config.get('num_actions', 7),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 4),
        num_tokens=config.get('num_tokens', 49),
        max_seq_len=config.get('max_seq_len', 64),
        dropout=config.get('dropout', 0.1),
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def check_state_format(test_trajectories, model, device='cpu'):
    """Check if state format matches between training and inference."""
    print("\n" + "="*60)
    print("CHECKING STATE FORMAT CONSISTENCY")
    print("="*60)
    
    # Get sample trajectory
    traj = test_trajectories[0]
    
    # Training format (from dataset)
    print("\n1. Training format (from dataset):")
    state_train = {
        'grid': torch.tensor([s['grid'] for s in traj['states']], dtype=torch.float32).unsqueeze(0),  # [1, seq_len, 147]
        'direction': torch.tensor([[s['direction'] for s in traj['states']]], dtype=torch.long),  # [1, seq_len]
    }
    print(f"   grid shape: {state_train['grid'].shape}")  # [1, seq_len, 147]
    print(f"   direction shape: {state_train['direction'].shape}")  # [1, seq_len]
    
    # Inference format (from test script)
    print("\n2. Inference format (from test script):")
    first_state = traj['states'][0]
    # Assume obs['image'] is [7, 7, 3] numpy array
    # Convert flattened grid back to spatial format
    grid_flat = np.array(first_state['grid'])  # [147]
    grid_spatial = grid_flat.reshape(7, 7, 3)  # [7, 7, 3]
    
    state_inference = {
        'grid': torch.tensor(grid_spatial, dtype=torch.float32).unsqueeze(0),  # [1, 7, 7, 3]
        'direction': torch.tensor([first_state['direction']], dtype=torch.long),  # [1]
    }
    print(f"   grid shape: {state_inference['grid'].shape}")  # [1, 7, 7, 3]
    print(f"   direction shape: {state_inference['direction'].shape}")  # [1]
    
    # Check if StateEncoder handles both formats correctly
    print("\n3. Testing StateEncoder with both formats...")
    state_encoder = model.state_encoder
    
    try:
        tokens_train = state_encoder(state_train)  # Should use first state
        print(f"   ✓ Training format works: output shape {tokens_train.shape}")
    except Exception as e:
        print(f"   ✗ Training format failed: {e}")
        tokens_train = None
    
    try:
        tokens_inference = state_encoder(state_inference)
        print(f"   ✓ Inference format works: output shape {tokens_inference.shape}")
    except Exception as e:
        print(f"   ✗ Inference format failed: {e}")
        tokens_inference = None
    
    # Check if outputs match (should be similar since both use first state)
    if tokens_train is not None and tokens_inference is not None:
        diff = torch.abs(tokens_train - tokens_inference).mean().item()
        print(f"\n   Token difference (train vs inference): {diff:.6f}")
        if diff < 1e-5:
            print("   ✓ States match! Format is consistent.")
        else:
            print(f"   ⚠ States differ! This might cause inference issues.")

def check_action_predictions(test_trajectories, model, device='cpu', num_samples=5):
    """Check model action predictions on validation set."""
    print("\n" + "="*60)
    print("CHECKING ACTION PREDICTIONS")
    print("="*60)
    
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(min(num_samples, len(test_trajectories))):
        traj = test_trajectories[i]
        
        # Prepare state (inference format)
        first_state = traj['states'][0]
        grid_flat = np.array(first_state['grid'])  # [147]
        grid_spatial = grid_flat.reshape(7, 7, 3)  # [7, 7, 3]
        
        state = {
            'grid': torch.tensor(grid_spatial, dtype=torch.float32).unsqueeze(0).to(device),
            'direction': torch.tensor([first_state['direction']], dtype=torch.long).to(device),
        }
        
        # Get ground truth actions
        gt_actions = torch.tensor(traj['actions'], dtype=torch.long)
        seq_len = len(gt_actions)
        
        # Model prediction (simplified: use t=0 directly)
        with torch.no_grad():
            # Initialize with mean action
            init_actions = torch.full((1, model.num_tokens), 2, dtype=torch.long, device=device)
            init_hidden = model.action_encoder(init_actions)
            
            # Add noise
            noise = torch.randn_like(init_hidden)
            hidden_state = init_hidden + 0.5 * noise
            
            # Denoise a few steps
            for step in range(10):
                t = 1.0 - (step + 1) / 10
                t_tensor = torch.tensor([t], device=device)
                hidden_state = model.denoise_step(hidden_state, state, t_tensor, guidance_scale=1.0)
            
            # Final decode
            t_final = torch.tensor([0.0], device=device)
            logits = model.forward(hidden_state, state, t_final)
            pred_actions = logits.argmax(dim=-1)[0]  # [num_tokens]
        
        # Compare with ground truth (only first seq_len tokens)
        pred_seq = pred_actions[:seq_len].cpu()
        
        matches = (pred_seq == gt_actions).sum().item()
        correct_predictions += matches
        total_predictions += seq_len
        
        if i < 3:  # Print details for first 3
            print(f"\nSample {i+1}:")
            print(f"  GT:   {gt_actions.tolist()[:15]}")
            print(f"  Pred: {pred_seq.tolist()[:15]}")
            print(f"  Matches: {matches}/{seq_len} ({100*matches/seq_len:.1f}%)")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    print(f"\nOverall token accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    print(f"  (Compare with training accuracy: ~84.5%)")
    
    if accuracy < 0.5:
        print("  ⚠ Very low accuracy! There's likely an inference bug.")
    elif accuracy < 0.7:
        print("  ⚠ Lower than training accuracy - possible format mismatch or denoising issue.")
    else:
        print("  ✓ Accuracy looks reasonable.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug training vs inference mismatch")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model_epoch37.pt',
                       help='Model checkpoint path')
    parser.add_argument('--data', type=str, default='data/FourRooms_test.pkl',
                       help='Test trajectory data path')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu, cuda, mps)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to check')
    
    args = parser.parse_args()
    
    # Load model
    model, config = load_model(args.checkpoint, device=args.device)
    
    # Load test trajectories
    print(f"\nLoading test trajectories from {args.data}...")
    with open(args.data, 'rb') as f:
        test_trajectories = pickle.load(f)
    print(f"Loaded {len(test_trajectories)} trajectories")
    
    # Run checks
    check_state_format(test_trajectories, model, device=args.device)
    check_action_predictions(test_trajectories, model, device=args.device, num_samples=args.num_samples)
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
