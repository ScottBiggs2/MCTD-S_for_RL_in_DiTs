"""
Utilities for verifying tensor shapes throughout the pipeline.
Critical for ensuring DiT training compatibility.
"""
import torch
from typing import Dict, List, Tuple, Optional


def verify_state_shape(state: Dict[str, torch.Tensor], expected_grid_size: int = 7) -> bool:
    """
    Verify state dictionary has correct shapes.
    
    Args:
        state: State dictionary from MazeEnvironment
        expected_grid_size: Expected grid size (default 7 for MiniGrid)
        
    Returns:
        True if shapes are correct
        
    Raises:
        AssertionError if shapes don't match
    """
    assert 'grid' in state, "State must contain 'grid' key"
    assert 'direction' in state, "State must contain 'direction' key"
    
    # Grid should be flattened: (grid_size * grid_size * 3,)
    expected_grid_dim = expected_grid_size * expected_grid_size * 3
    assert state['grid'].shape == (expected_grid_dim,), \
        f"Grid shape {state['grid'].shape} != ({expected_grid_dim},)"
    
    # Direction should be scalar or (1,)
    assert state['direction'].shape == () or state['direction'].shape == (1,), \
        f"Direction shape {state['direction'].shape} should be scalar or (1,)"
    
    # Position is optional
    if 'position' in state and state['position'] is not None:
        assert state['position'].shape == (2,), \
            f"Position shape {state['position'].shape} != (2,)"
    
    return True


def verify_action_sequence_shape(actions: torch.Tensor, max_seq_len: int) -> bool:
    """
    Verify action sequence has correct shape.
    
    Args:
        actions: Action tensor [seq_len] or [batch, seq_len]
        max_seq_len: Maximum sequence length
        
    Returns:
        True if shapes are correct
    """
    assert len(actions.shape) in [1, 2], \
        f"Actions should be 1D or 2D, got shape {actions.shape}"
    
    if len(actions.shape) == 1:
        seq_len = actions.shape[0]
    else:
        seq_len = actions.shape[1]
    
    assert seq_len <= max_seq_len, \
        f"Sequence length {seq_len} exceeds max_seq_len {max_seq_len}"
    
    return True


def verify_hidden_action_shape(hidden_actions: torch.Tensor, seq_len: int, hidden_dim: int) -> bool:
    """
    Verify hidden action embeddings have correct shape for DiT.
    
    Args:
        hidden_actions: Hidden action tensor [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        seq_len: Expected sequence length
        hidden_dim: Expected hidden dimension
        
    Returns:
        True if shapes are correct
    """
    assert len(hidden_actions.shape) in [2, 3], \
        f"Hidden actions should be 2D or 3D, got shape {hidden_actions.shape}"
    
    if len(hidden_actions.shape) == 2:
        # [seq_len, hidden_dim]
        assert hidden_actions.shape == (seq_len, hidden_dim), \
            f"Hidden actions shape {hidden_actions.shape} != ({seq_len}, {hidden_dim})"
    else:
        # [batch, seq_len, hidden_dim]
        batch_size = hidden_actions.shape[0]
        assert hidden_actions.shape == (batch_size, seq_len, hidden_dim), \
            f"Hidden actions shape {hidden_actions.shape} != ({batch_size}, {seq_len}, {hidden_dim})"
    
    return True


def verify_cnn_token_shape(tokens: torch.Tensor, num_tokens: int, hidden_dim: int, batch_size: int = None) -> bool:
    """
    Verify CNN tokenized state has correct shape for DiT.
    
    Args:
        tokens: Token tensor [batch, num_tokens, hidden_dim] or [num_tokens, hidden_dim]
        num_tokens: Expected number of tokens (e.g., 49 for 7x7 grid)
        hidden_dim: Expected hidden dimension
        batch_size: Expected batch size (if None, checks any batch size)
        
    Returns:
        True if shapes are correct
    """
    assert len(tokens.shape) in [2, 3], \
        f"Tokens should be 2D or 3D, got shape {tokens.shape}"
    
    if len(tokens.shape) == 2:
        # [num_tokens, hidden_dim]
        assert tokens.shape == (num_tokens, hidden_dim), \
            f"Tokens shape {tokens.shape} != ({num_tokens}, {hidden_dim})"
    else:
        # [batch, num_tokens, hidden_dim]
        if batch_size is not None:
            assert tokens.shape == (batch_size, num_tokens, hidden_dim), \
                f"Tokens shape {tokens.shape} != ({batch_size}, {num_tokens}, {hidden_dim})"
        else:
            assert tokens.shape[1] == num_tokens and tokens.shape[2] == hidden_dim, \
                f"Tokens shape {tokens.shape} should have ({num_tokens}, {hidden_dim}) for last two dims"
    
    return True


def verify_batch_shapes(
    states: Dict[str, torch.Tensor],
    actions: torch.Tensor,
    hidden_actions: torch.Tensor,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    grid_size: int = 7
) -> Dict[str, bool]:
    """
    Verify all batch shapes are consistent for DiT training.
    
    Args:
        states: Batched state dictionary
        actions: Action sequences [batch, seq_len]
        hidden_actions: Hidden action embeddings [batch, seq_len, hidden_dim]
        batch_size: Expected batch size
        seq_len: Expected sequence length
        hidden_dim: Expected hidden dimension
        grid_size: Grid size for state verification
        
    Returns:
        Dictionary of verification results
    """
    results = {}
    
    # Verify batched states
    if 'grid' in states:
        expected_grid_dim = grid_size * grid_size * 3
        results['state_grid'] = states['grid'].shape == (batch_size, expected_grid_dim)
    
    if 'direction' in states:
        results['state_direction'] = states['direction'].shape == (batch_size,) or \
                                     states['direction'].shape == (batch_size, 1)
    
    # Verify actions
    results['actions'] = actions.shape == (batch_size, seq_len)
    
    # Verify hidden actions
    results['hidden_actions'] = hidden_actions.shape == (batch_size, seq_len, hidden_dim)
    
    return results


def print_shape_summary(
    states: Dict[str, torch.Tensor],
    actions: Optional[torch.Tensor] = None,
    hidden_actions: Optional[torch.Tensor] = None
):
    """
    Print a summary of tensor shapes for debugging.
    
    Args:
        states: State dictionary
        actions: Optional action tensor
        hidden_actions: Optional hidden action tensor
    """
    print("=" * 50)
    print("Shape Verification Summary")
    print("=" * 50)
    
    print("\nStates:")
    for key, value in states.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    if actions is not None:
        print(f"\nActions: {actions.shape}")
    
    if hidden_actions is not None:
        print(f"Hidden Actions: {hidden_actions.shape}")
    
    print("=" * 50)
