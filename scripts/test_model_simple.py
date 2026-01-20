"""
Simple test to check if model produces identical outputs for all tokens.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from src.models.diffusion_policy import DiffusionPolicy


def main():
    print("Testing if model produces identical outputs...")
    
    config = {
        'num_actions': 7,
        'hidden_dim': 128,
        'num_layers': 6,
        'num_heads': 4,
        'num_tokens': 361,
        'max_seq_len': 32,
        'dropout': 0.1,
    }
    
    model = DiffusionPolicy(**config)
    model.eval()
    
    B = 1
    num_tokens = 49
    hidden_dim = config['hidden_dim']
    
    # Create INTENTIONALLY DIFFERENT input tokens
    noisy_actions = torch.randn(B, num_tokens, hidden_dim)
    for i in range(num_tokens):
        noisy_actions[0, i, :] = torch.randn(hidden_dim) * (i + 1)  # Each token is unique
    
    state = {
        'grid': torch.randn(B, 7, 7, 3),
        'direction': torch.randint(0, 4, (B,)),
    }
    
    t = torch.tensor([0.0])
    
    print(f"Input tokens are different:")
    print(f"  Token 0 mean: {noisy_actions[0, 0].mean():.4f}")
    print(f"  Token 10 mean: {noisy_actions[0, 10].mean():.4f}")
    print(f"  Token 48 mean: {noisy_actions[0, 48].mean():.4f}")
    
    with torch.no_grad():
        logits = model.forward(noisy_actions, state, t)
    
    print(f"\nOutput logits:")
    print(f"  Token 0: {logits[0, 0].tolist()}")
    print(f"  Token 10: {logits[0, 10].tolist()}")
    print(f"  Token 48: {logits[0, 48].tolist()}")
    
    # Check if all tokens are identical
    all_same = True
    for i in range(1, num_tokens):
        diff = torch.abs(logits[0, 0] - logits[0, i])
        if diff.max() > 1e-5:
            all_same = False
            print(f"\n✅ Tokens are DIFFERENT (token 0 vs {i}, max_diff={diff.max():.4f})")
            break
    
    if all_same:
        print(f"\n❌ PROBLEM: All tokens produce IDENTICAL outputs!")
        print("   This means the model architecture is collapsing token diversity.")
        print("   Possible causes:")
        print("   1. Missing positional embeddings for action tokens")
        print("   2. DiT blocks are collapsing all tokens")
        print("   3. Training learned a degenerate solution")
    else:
        print(f"\n✅ Model architecture preserves token diversity")


if __name__ == '__main__':
    main()
