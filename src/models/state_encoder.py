"""
State encoder using CNN tokenizer with position embeddings.

Architecture: State -> CNN Tokenizer -> tokens + pos_embed
"""
import torch
import torch.nn as nn
from typing import Dict
from .components.cnn_tokenizer import StateCNNTokenizer
from .components.position_embedding import PositionalEmbedding


class StateEncoder(nn.Module):
    """
    Encode MiniGrid observations into tokenized representations with position embeddings.
    
    Uses CNN tokenizer to convert grid observations to tokens,
    then adds positional embeddings for DiT processing.
    """
    def __init__(
        self,
        grid_size=7,
        hidden_dim=128,
        num_tokens=None,  # If None, uses grid_size * grid_size
        use_learned_pos=True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        
        if num_tokens is None:
            num_tokens = grid_size * grid_size
        
        self.num_tokens = num_tokens
        
        # CNN tokenizer for grid
        self.grid_tokenizer = StateCNNTokenizer(
            grid_size=grid_size,
            num_channels=3,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
        )
        
        # Direction embedding (separate from grid)
        self.direction_embed = nn.Embedding(4, hidden_dim)
        
        # Position embeddings
        if use_learned_pos:
            self.pos_embed = PositionalEmbedding(num_tokens, hidden_dim)
        else:
            from .components.position_embedding import SinusoidalPositionalEmbedding
            self.pos_embed = SinusoidalPositionalEmbedding(num_tokens, hidden_dim)
        
        # Optional: combine direction with tokens
        # We can either:
        # 1. Add direction to all tokens
        # 2. Concatenate direction as an additional token
        # 3. Use direction to modulate tokens
        # For now, we'll add it to all tokens (simple and effective)
        self.direction_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode state to tokenized representation with position embeddings.
        
        Args:
            state_dict: Dictionary with:
                - 'grid': (batch, grid_size, grid_size, 3) or (batch, seq_len, 147) or (batch, 147) or (batch, 3, grid_size, grid_size)
                - 'direction': (batch,) or (batch, seq_len) or (batch, 1) agent direction (0-3)
        
        Returns:
            tokens: (batch, num_tokens, hidden_dim) with position embeddings added
        """
        grid = state_dict['grid']
        direction = state_dict['direction']
        
        # Handle different input formats
        # Case 1: Flattened grid [B, seq_len, 147] - take first state and reshape
        # Case 2: Flattened grid [B, 147] - reshape to [B, 7, 7, 3]
        # Case 3: Already shaped [B, 7, 7, 3] or [B, 3, 7, 7]
        
        # Detect grid size dynamically from input shape
        if grid.dim() == 3:
            # [B, seq_len, H*W*3] - take first state for initial condition
            B, seq_len, flat_size = grid.shape
            # Infer grid size from flattened size: H*W*3 -> sqrt(H*W) = grid_size
            inferred_size = int((flat_size // 3) ** 0.5)
            if inferred_size * inferred_size * 3 != flat_size:
                raise ValueError(
                    f"Cannot infer grid size from flat_size={flat_size}. "
                    f"Expected H*W*3 where H=W. Got inferred_size={inferred_size}, "
                    f"which gives {inferred_size * inferred_size * 3} (expected {flat_size})"
                )
            # Use inferred size or fallback to self.grid_size if mismatch
            actual_grid_size = inferred_size
            
            # Take only first state (initial condition for diffusion)
            # Make contiguous before reshape
            grid = grid[:, 0, :].contiguous()  # [B, H*W*3]
            # Reshape to spatial format: [B, H*W*3] -> [B, H, W, 3]
            grid = grid.reshape(B, actual_grid_size, actual_grid_size, 3)
            
            # Update self.grid_size if different (for CNN tokenizer compatibility)
            if actual_grid_size != self.grid_size:
                # Warn but allow it (for now) - we'll update CNN tokenizer to handle this
                # TODO: Make CNN tokenizer handle variable grid sizes
                pass
            
            # Take first direction too
            if direction.dim() == 2:
                direction = direction[:, 0]  # [B]
            elif direction.dim() == 1:
                if direction.shape[0] == B * seq_len:
                    # Reshape to [B, seq_len] then take first
                    direction = direction.reshape(B, seq_len)[:, 0]
        elif grid.dim() == 2:
            # [B, H*W*3] - single flattened state
            B, flat_size = grid.shape
            # Infer grid size from flattened size
            inferred_size = int((flat_size // 3) ** 0.5)
            if inferred_size * inferred_size * 3 != flat_size:
                raise ValueError(
                    f"Cannot infer grid size from flat_size={flat_size}. "
                    f"Expected H*W*3 where H=W. Got inferred_size={inferred_size}, "
                    f"which gives {inferred_size * inferred_size * 3} (expected {flat_size})"
                )
            actual_grid_size = inferred_size
            
            # Make contiguous before reshape
            grid = grid.contiguous().reshape(B, actual_grid_size, actual_grid_size, 3)
            
            # Update self.grid_size if different
            if actual_grid_size != self.grid_size:
                pass  # TODO: Update CNN tokenizer to handle variable sizes
        elif grid.dim() == 4:
            # Already in spatial format: [B, H, W, C] or [B, C, H, W]
            if grid.shape[1] == 3 or grid.shape[-1] == 3:
                # [B, 3, H, W] or [B, H, W, 3] - good to go
                pass
            else:
                raise ValueError(f"Unexpected grid shape: {grid.shape}")
        else:
            raise ValueError(f"Unexpected grid dimensions: {grid.dim()}, shape: {grid.shape}")
        
        # Ensure direction is 1D and matches batch size
        # Handle direction: might be [B, seq_len], [B, 1], or [B]
        if direction.dim() == 2:
            # [B, seq_len] - take first direction
            direction = direction[:, 0]  # [B]
        elif direction.dim() == 1:
            # [B] or [B*seq_len]
            if direction.shape[0] == grid.shape[0] * (direction.shape[0] // grid.shape[0]) and direction.shape[0] != grid.shape[0]:
                # Flattened sequence - take first element of each sample
                seq_len = direction.shape[0] // grid.shape[0]
                direction = direction.reshape(grid.shape[0], seq_len)[:, 0]
            # Otherwise already [B], keep as is
        elif direction.dim() == 0:
            # Scalar - expand to batch
            direction = direction.expand(grid.shape[0])
        
        # Final check: ensure direction matches grid batch size
        if direction.shape[0] != grid.shape[0]:
            raise ValueError(
                f"Direction shape {direction.shape} doesn't match grid batch size {grid.shape[0]}. "
                f"Grid shape: {grid.shape}, Direction shape: {direction.shape}"
            )
        
        # Tokenize grid
        grid_tokens = self.grid_tokenizer(grid)  # (batch, num_tokens, hidden_dim)
        
        # Encode direction
        dir_embed = self.direction_embed(direction)  # (batch, hidden_dim)
        dir_embed = self.direction_proj(dir_embed)  # (batch, hidden_dim)
        
        # Add direction to all tokens (broadcast)
        tokens = grid_tokens + dir_embed.unsqueeze(1)  # (batch, num_tokens, hidden_dim)
        
        # Add positional embeddings
        tokens = self.pos_embed(tokens)  # (batch, num_tokens, hidden_dim)
        
        return tokens
