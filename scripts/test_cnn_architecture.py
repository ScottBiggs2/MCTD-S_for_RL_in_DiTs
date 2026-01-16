"""
Test CNN tokenizer architecture and verify shapes for DiT training.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import minigrid to register environments
try:
    import minigrid
except ImportError:
    print("ERROR: minigrid package not found. Install with: pip install minigrid")
    sys.exit(1)

import torch
from src.models.state_encoder import StateEncoder
from src.models.action_encoder import ActionEncoder, ActionDecoder
from src.environments.minigrid_wrapper import MazeEnvironment
from src.utils.shape_verification import print_shape_summary


def test_state_encoder():
    """Test StateEncoder with CNN tokenizer."""
    print("=" * 60)
    print("Testing StateEncoder (CNN Tokenizer)")
    print("=" * 60)
    
    # Initialize encoder
    encoder = StateEncoder(
        grid_size=7,
        hidden_dim=128,
        num_tokens=49,  # 7x7
        use_learned_pos=True,
    )
    
    # Create test state
    env = MazeEnvironment("MiniGrid-Empty-8x8-v0")
    state = env.reset()
    
    # Convert to batch format
    batch_size = 4
    batch_states = {
        'grid': torch.stack([state['grid'].reshape(7, 7, 3) for _ in range(batch_size)]),
        'direction': torch.stack([state['direction'] for _ in range(batch_size)]),
    }
    
    print("\nInput shapes:")
    print(f"  grid: {batch_states['grid'].shape}")
    print(f"  direction: {batch_states['direction'].shape}")
    
    # Forward pass
    tokens = encoder(batch_states)
    
    print(f"\nOutput tokens shape: {tokens.shape}")
    print(f"Expected: (batch_size={batch_size}, num_tokens=49, hidden_dim=128)")
    
    assert tokens.shape == (batch_size, 49, 128), \
        f"Token shape {tokens.shape} != (4, 49, 128)"
    
    print("✓ StateEncoder test passed!")
    env.close()
    return True


def test_action_encoder():
    """Test ActionEncoder."""
    print("\n" + "=" * 60)
    print("Testing ActionEncoder")
    print("=" * 60)
    
    encoder = ActionEncoder(num_actions=7, hidden_dim=128)
    
    batch_size = 4
    seq_len = 10
    actions = torch.randint(0, 7, (batch_size, seq_len))
    
    print(f"\nInput actions shape: {actions.shape}")
    
    # Encode
    hidden = encoder(actions)
    print(f"Encoded hidden shape: {hidden.shape}")
    print(f"Expected: (batch_size={batch_size}, seq_len={seq_len}, hidden_dim=128)")
    
    assert hidden.shape == (batch_size, seq_len, 128), \
        f"Hidden shape {hidden.shape} != (4, 10, 128)"
    
    # Decode
    logits = encoder.decode(hidden)
    print(f"Decoded logits shape: {logits.shape}")
    print(f"Expected: (batch_size={batch_size}, seq_len={seq_len}, num_actions=7)")
    
    assert logits.shape == (batch_size, seq_len, 7), \
        f"Logits shape {logits.shape} != (4, 10, 7)"
    
    print("✓ ActionEncoder test passed!")
    return True


def test_action_decoder():
    """Test ActionDecoder with CNN tokenizer."""
    print("\n" + "=" * 60)
    print("Testing ActionDecoder (CNN Tokenizer)")
    print("=" * 60)
    
    decoder = ActionDecoder(
        hidden_dim=128,
        num_tokens=49,
        num_actions=7,
        grid_size=7,
    )
    
    batch_size = 4
    tokens = torch.randn(batch_size, 49, 128)  # From DiT
    
    print(f"\nInput tokens shape: {tokens.shape}")
    
    # Decode
    action_logits = decoder(tokens)
    print(f"Output action logits shape: {action_logits.shape}")
    print(f"Expected: (batch_size={batch_size}, num_tokens=49, num_actions=7)")
    
    assert action_logits.shape == (batch_size, 49, 7), \
        f"Action logits shape {action_logits.shape} != (4, 49, 7)"
    
    # Test sequence decoding
    seq_logits = decoder.forward_to_sequence(tokens, seq_len=64)
    print(f"Sequence logits shape: {seq_logits.shape}")
    print(f"Expected: (batch_size={batch_size}, seq_len=64, num_actions=7)")
    
    assert seq_logits.shape == (batch_size, 64, 7), \
        f"Sequence logits shape {seq_logits.shape} != (4, 64, 7)"
    
    print("✓ ActionDecoder test passed!")
    return True


def test_end_to_end_shapes():
    """Test end-to-end shape compatibility."""
    print("\n" + "=" * 60)
    print("Testing End-to-End Shape Compatibility")
    print("=" * 60)
    
    # Initialize components
    state_encoder = StateEncoder(hidden_dim=128, num_tokens=49)
    action_decoder = ActionDecoder(hidden_dim=128, num_tokens=49, num_actions=7)
    
    # Create batch
    batch_size = 4
    env = MazeEnvironment("MiniGrid-Empty-8x8-v0")
    state = env.reset()
    
    batch_states = {
        'grid': torch.stack([state['grid'].reshape(7, 7, 3) for _ in range(batch_size)]),
        'direction': torch.stack([state['direction'] for _ in range(batch_size)]),
    }
    
    # Encode state
    state_tokens = state_encoder(batch_states)  # (B, 49, 128)
    print(f"State tokens: {state_tokens.shape}")
    
    # Simulate DiT output (same shape as input for now)
    dit_output = state_tokens  # (B, 49, 128)
    print(f"DiT output: {dit_output.shape}")
    
    # Decode to actions
    action_logits = action_decoder.forward_to_sequence(dit_output, seq_len=64)  # (B, 64, 7)
    print(f"Action logits: {action_logits.shape}")
    
    # Verify shapes
    assert state_tokens.shape == (batch_size, 49, 128), "State tokens shape incorrect"
    assert dit_output.shape == (batch_size, 49, 128), "DiT output shape incorrect"
    assert action_logits.shape == (batch_size, 64, 7), "Action logits shape incorrect"
    
    print("\n✓ All shapes compatible for DiT training!")
    env.close()
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CNN Architecture Shape Verification")
    print("=" * 60)
    
    all_passed = True
    
    try:
        all_passed &= test_state_encoder()
        all_passed &= test_action_encoder()
        all_passed &= test_action_decoder()
        all_passed &= test_end_to_end_shapes()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exception(*sys.exc_info())
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - CNN architecture ready!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
