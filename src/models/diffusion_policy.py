"""
Diffusion Policy for action sequences.

Masked Diffusion Language Model operating on continuous hidden representations.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .state_cnn_encoder import StateCNNEncoder
from .action_encoder import ActionEncoder
from .components.timestep_embed import SinusoidalPositionEmbedding
from .components.dit_block import DiTBlock
from .components.position_embedding import PositionalEmbedding


class DiffusionPolicy(nn.Module):
    """
    Masked Diffusion Language Model for action sequences.
    
    Operates on continuous hidden action representations (from CNN tokenizers),
    denoises to discrete action logits.
    
    Architecture:
        State (19x19 grid) -> StateEncoder -> [B, 361, 128] tokens -> POOLED to [B, 128] conditioning
        Noisy Actions -> [B, 32, 128] hidden (max_seq_len actions, NOT num_tokens!)
        + Timestep embedding [B, 128]
        + State conditioning [B, 128]
        ↓
        DiT Blocks (num_layers)
        ↓
        Action Logits [B, 32, 7]
    
    IMPORTANT: 
    - num_tokens (361) is for STATE encoding (19x19 grid cells) → pooled to conditioning vector
    - max_seq_len (32) is for ACTION sequences → these are what we predict
    - MiniGrid actions: 0=turn_left, 1=turn_right, 2=move_forward, 3=pickup, 4=drop, 5=toggle, 6=done
    """
    def __init__(
        self,
        num_actions=7,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        num_tokens=49,  # Default: 7x7 grid, but can be larger (e.g., 361 for 19x19)
        max_seq_len=64,
        dropout=0.1,
        grid_size=None,  # If None, inferred from num_tokens (assumes square grid)
    ):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        
        # Infer grid_size from num_tokens if not provided
        if grid_size is None:
            grid_size = int(num_tokens ** 0.5)
            if grid_size * grid_size != num_tokens:
                raise ValueError(
                    f"Cannot infer grid_size from num_tokens={num_tokens}. "
                    f"Expected num_tokens to be a perfect square, got {num_tokens}"
                )
        self.grid_size = grid_size
        
        # State encoder: Simple CNN that outputs single embedding vector (CLIP-style)
        self.state_encoder = StateCNNEncoder(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_channels=3,
        )
        
        # Action encoder for converting discrete actions to hidden space
        self.action_encoder = ActionEncoder(num_actions, hidden_dim)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # State conditioning: StateCNNEncoder directly outputs [B, hidden_dim]
        # No projection needed - encoder already outputs single vector (CLIP-style)
        
        # Positional embeddings for action tokens
        # CRITICAL: Without this, model can't distinguish token positions!
        # IMPORTANT: Use max_seq_len (not num_tokens) - we predict action sequences, not state tokens!
        # num_tokens is for state encoding (361 grid cells), but actions are a sequence (32 actions)
        self.action_pos_embed = PositionalEmbedding(
            num_tokens=max_seq_len,  # Action sequence length, not state grid size!
            hidden_dim=hidden_dim
        )
        
        # DiT transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.final_norm = nn.LayerNorm(hidden_dim)
        # Output action logits per token
        self.action_head = nn.Linear(hidden_dim, num_actions)
    
    def forward(
        self,
        noisy_actions: torch.Tensor,
        state: Dict[str, torch.Tensor],
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Denoise action sequence conditioned on state and timestep.
        
        IMPORTANT: 
        - State encoding produces num_tokens (361 for 19x19 grid) which are POOLED to conditioning
        - Action prediction produces max_seq_len (32) actions, NOT num_tokens actions
        
        Args:
            noisy_actions: [B, max_seq_len, hidden_dim] continuous hidden action states
            state: dict with 'grid' [B, H, W, 3] (e.g., [B, 19, 19, 3]) and 'direction' [B]
            t: [B] diffusion timestep in [0, 1]
            mask: [B, max_seq_len] binary mask (1 = masked position), optional
        
        Returns:
            [B, max_seq_len, num_actions] denoised action logits
        """
        B = noisy_actions.shape[0]
        
        # Encode state to single embedding vector (CLIP-style conditioning)
        # StateCNNEncoder directly outputs [B, hidden_dim] - no pooling needed
        state_cond = self.state_encoder(state)  # [B, hidden_dim]
        
        # Encode timestep
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        
        # Combine state and time conditioning
        cond_emb = state_cond + t_emb  # [B, hidden_dim]
        
        # Input is already tokenized (noisy_actions are hidden representations)
        # IMPORTANT: noisy_actions should be [B, max_seq_len, hidden_dim], NOT [B, num_tokens, hidden_dim]
        x = noisy_actions  # [B, max_seq_len, hidden_dim]
        
        # CRITICAL: Add positional embeddings so model can distinguish token positions
        # Without this, all tokens look identical and model outputs same action everywhere
        # Positional embedding uses max_seq_len (action sequence length), not num_tokens (state grid size)
        x = self.action_pos_embed(x)  # [B, max_seq_len, hidden_dim]
        
        # Apply DiT blocks with conditioning
        for block in self.blocks:
            x = block(x, cond_emb)
        
        # Output head
        x = self.final_norm(x)
        logits = self.action_head(x)  # [B, max_seq_len, num_actions]
        
        return logits
    
    def denoise_step(
        self,
        hidden_actions: torch.Tensor,
        state: Dict[str, torch.Tensor],
        t: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Single denoising step (for MCTD inference).
        
        Args:
            hidden_actions: [B, max_seq_len, hidden_dim] (action sequence, NOT state tokens!)
            state: state dict (grid is encoded to num_tokens=361, then pooled to conditioning)
            t: [B] current noise level in [0, 1]
            guidance_scale: float, classifier-free guidance strength
        
        Returns:
            [B, max_seq_len, hidden_dim] less noisy hidden actions
        """
        # Forward pass to get action logits
        logits = self.forward(hidden_actions, state, t)  # [B, max_seq_len, num_actions]
        
        # Get predicted clean actions
        pred_actions = logits.argmax(dim=-1)  # [B, max_seq_len]
        
        # Encode to hidden space
        clean_hidden = self.action_encoder(pred_actions)  # [B, max_seq_len, hidden_dim]
        
        # Denoising update: interpolate toward clean prediction
        # More clean as t → 0
        alpha = 1.0 - t[:, None, None]  # [B, 1, 1]
        denoised = alpha * clean_hidden + (1 - alpha) * hidden_actions
        
        # Optional: classifier-free guidance
        if guidance_scale != 1.0:
            # Compute unconditional prediction (masked state)
            # For unconditional, we could mask the state or use zeros
            # Simplified: use masked state
            masked_state = {
                'grid': torch.zeros_like(state['grid']),
                'direction': torch.zeros_like(state['direction']),
            }
            uncond_logits = self.forward(hidden_actions, masked_state, t)
            uncond_actions = uncond_logits.argmax(dim=-1)
            uncond_hidden = self.action_encoder(uncond_actions)
            
            # Guidance: move away from uncond toward cond
            denoised = uncond_hidden + guidance_scale * (denoised - uncond_hidden)
        
        return denoised
