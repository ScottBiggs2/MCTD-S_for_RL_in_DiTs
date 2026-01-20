"""
Test model architecture to identify why all tokens produce identical outputs.

Tests:
1. Fresh model with random initialization
2. Tokenizer output shapes and diversity
3. Model forward pass with toy inputs
4. Check if action_head is collapsing tokens
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np

from src.models.diffusion_policy import DiffusionPolicy
from src.models.components.cnn_tokenizer import ActionCNNTokenizer


def test_action_head_directly():
    """Test action_head (linear layer) directly with diverse inputs."""
    print("="*60)
    print("TEST 1: Action Head (Linear Layer) Direct Test")
    print("="*60)
    
    hidden_dim = 64
    num_tokens = 49
    num_actions = 7
    
    # Create a simple linear action head
    action_head = nn.Linear(hidden_dim, num_actions)
    
    # Create diverse input tokens (each token is different)
    B = 1
    tokens = torch.randn(B, num_tokens, hidden_dim)
    
    # Make tokens intentionally different
    for i in range(num_tokens):
        tokens[0, i, :] = torch.randn(hidden_dim) * (i + 1)  # Each token is unique
    
    print(f"Input tokens shape: {tokens.shape}")
    print(f"Input token 0 mean: {tokens[0, 0].mean():.4f}, std: {tokens[0, 0].std():.4f}")
    print(f"Input token 10 mean: {tokens[0, 10].mean():.4f}, std: {tokens[0, 10].std():.4f}")
    print(f"Input token 48 mean: {tokens[0, 48].mean():.4f}, std: {tokens[0, 48].std():.4f}")
    
    # Forward pass
    logits = action_head(tokens)  # [B, num_tokens, num_actions]
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits token 0: {logits[0, 0].tolist()}")
    print(f"Logits token 10: {logits[0, 10].tolist()}")
    print(f"Logits token 48: {logits[0, 48].tolist()}")
    
    # Check if outputs are identical
    logits_diff = torch.abs(logits[0, 0] - logits[0, 10])
    print(f"Difference between token 0 and 10: {logits_diff.tolist()}")
    print(f"Max difference: {logits_diff.max():.4f}")
    
    if logits_diff.max() < 1e-5:
        print("❌ PROBLEM: Outputs are identical!")
        return False
    else:
        print("✅ Outputs are diverse")
        return True


def test_action_cnn_tokenizer():
    """Test ActionCNNTokenizer with diverse inputs."""
    print("\n" + "="*60)
    print("TEST 2: ActionCNNTokenizer Test")
    print("="*60)
    
    hidden_dim = 64
    num_tokens = 49
    num_actions = 7
    grid_size = 7
    
    tokenizer = ActionCNNTokenizer(
        hidden_dim=hidden_dim,
        num_tokens=num_tokens,
        num_actions=num_actions,
        grid_size=grid_size,
    )
    
    # Create diverse input tokens
    B = 1
    tokens = torch.randn(B, num_tokens, hidden_dim)
    
    # Make tokens intentionally different
    for i in range(num_tokens):
        tokens[0, i, :] = torch.randn(hidden_dim) * (i + 1)
    
    print(f"Input tokens shape: {tokens.shape}")
    print(f"Input token 0 mean: {tokens[0, 0].mean():.4f}, std: {tokens[0, 0].std():.4f}")
    print(f"Input token 10 mean: {tokens[0, 10].mean():.4f}, std: {tokens[0, 10].std():.4f}")
    
    # Forward pass
    logits = tokenizer(tokens)  # [B, num_tokens, num_actions]
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits token 0: {logits[0, 0].tolist()}")
    print(f"Logits token 10: {logits[0, 10].tolist()}")
    print(f"Logits token 48: {logits[0, 48].tolist()}")
    
    # Check if outputs are identical
    logits_diff_01 = torch.abs(logits[0, 0] - logits[0, 1])
    logits_diff_048 = torch.abs(logits[0, 0] - logits[0, 48])
    print(f"Difference between token 0 and 1: max={logits_diff_01.max():.4f}")
    print(f"Difference between token 0 and 48: max={logits_diff_048.max():.4f}")
    
    # Check all tokens
    all_same = True
    for i in range(1, num_tokens):
        diff = torch.abs(logits[0, 0] - logits[0, i])
        if diff.max() > 1e-5:
            all_same = False
            break
    
    if all_same:
        print("❌ PROBLEM: All token outputs are identical!")
        print("   This is the bug! The ActionCNNTokenizer is collapsing tokens.")
        return False
    else:
        print("✅ Outputs are diverse")
        return True


def test_full_model_fresh():
    """Test full DiffusionPolicy model with fresh initialization."""
    print("\n" + "="*60)
    print("TEST 3: Full Model (Fresh) Test")
    print("="*60)
    
    config = {
        'num_actions': 7,
        'hidden_dim': 64,
        'num_layers': 4,
        'num_heads': 4,
        'num_tokens': 49,
        'max_seq_len': 64,
        'dropout': 0.1,
    }
    
    model = DiffusionPolicy(**config)
    model.eval()
    
    # Create diverse input
    B = 1
    num_tokens = 49
    hidden_dim = 64
    
    # Create diverse noisy actions
    noisy_actions = torch.randn(B, num_tokens, hidden_dim)
    for i in range(num_tokens):
        noisy_actions[0, i, :] = torch.randn(hidden_dim) * (i + 1)
    
    # Create dummy state
    state = {
        'grid': torch.randn(B, 7, 7, 3),
        'direction': torch.randint(0, 4, (B,)),
    }
    
    t = torch.tensor([0.0])
    
    print(f"Noisy actions shape: {noisy_actions.shape}")
    print(f"Noisy actions token 0 mean: {noisy_actions[0, 0].mean():.4f}")
    print(f"Noisy actions token 10 mean: {noisy_actions[0, 10].mean():.4f}")
    
    # Forward pass
    with torch.no_grad():
        logits = model.forward(noisy_actions, state, t)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Logits token 0: {logits[0, 0].tolist()}")
    print(f"Logits token 10: {logits[0, 10].tolist()}")
    print(f"Logits token 48: {logits[0, 48].tolist()}")
    
    # Check if outputs are identical
    logits_diff_01 = torch.abs(logits[0, 0] - logits[0, 1])
    logits_diff_048 = torch.abs(logits[0, 0] - logits[0, 48])
    print(f"Difference between token 0 and 1: max={logits_diff_01.max():.4f}")
    print(f"Difference between token 0 and 48: max={logits_diff_048.max():.4f}")
    
    # Check all tokens
    all_same = True
    max_diff = 0.0
    for i in range(1, num_tokens):
        diff = torch.abs(logits[0, 0] - logits[0, i])
        max_diff = max(max_diff, diff.max().item())
        if diff.max() > 1e-5:
            all_same = False
    
    if all_same:
        print(f"❌ PROBLEM: All token outputs are identical! (max_diff={max_diff:.6f})")
        print("   The model architecture is collapsing tokens.")
        return False
    else:
        print(f"✅ Outputs are diverse (max_diff={max_diff:.4f})")
        return True


def test_action_head_in_model():
    """Check what action_head actually is in the model."""
    print("\n" + "="*60)
    print("TEST 4: Inspect Model's action_head")
    print("="*60)
    
    config = {
        'num_actions': 7,
        'hidden_dim': 64,
        'num_layers': 4,
        'num_heads': 4,
        'num_tokens': 49,
        'max_seq_len': 64,
        'dropout': 0.1,
    }
    
    model = DiffusionPolicy(**config)
    
    print(f"action_head type: {type(model.action_head)}")
    print(f"action_head: {model.action_head}")
    
    # Check if it's a linear layer or something else
    if isinstance(model.action_head, nn.Linear):
        print("✅ action_head is a Linear layer (correct)")
        print(f"   Weight shape: {model.action_head.weight.shape}")
        print(f"   Bias shape: {model.action_head.bias.shape if model.action_head.bias is not None else 'None'}")
    else:
        print("❌ action_head is NOT a Linear layer!")
        print(f"   It's a {type(model.action_head)}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE DIAGNOSTIC TESTS")
    print("="*60)
    
    results = {}
    
    # Test 1: Action head directly
    results['action_head'] = test_action_head_directly()
    
    # Test 2: ActionCNNTokenizer
    results['tokenizer'] = test_action_cnn_tokenizer()
    
    # Test 3: Full model
    results['full_model'] = test_full_model_fresh()
    
    # Test 4: Inspect action_head
    results['inspect'] = test_action_head_in_model()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    # Identify the problem
    if not results.get('tokenizer', True):
        print("\n🔍 ROOT CAUSE: ActionCNNTokenizer is collapsing tokens!")
        print("   The CNN tokenizer is not preserving per-token information.")
    elif not results.get('full_model', True):
        print("\n🔍 ROOT CAUSE: Full model is collapsing tokens!")
        print("   Check the DiT blocks or action_head integration.")
    else:
        print("\n✅ Architecture looks correct. Problem might be in training.")


if __name__ == '__main__':
    main()
