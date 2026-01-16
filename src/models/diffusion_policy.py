"""
Diffusion Policy for action sequences.

Masked Diffusion Language Model operating on continuous hidden representations.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .state_encoder import StateEncoder
from .action_encoder import ActionEncoder
from .components.timestep_embed import SinusoidalPositionEmbedding
from .components.dit_block import DiTBlock


class DiffusionPolicy(nn.Module):
    """
    Masked Diffusion Language Model for action sequences.
    
    Operates on continuous hidden action representations (from CNN tokenizers),
    denoises to discrete action logits.
    
    Architecture:
        State -> StateEncoder -> [B, 49, 128] tokens
        Noisy Actions -> [B, 49, 128] hidden
        + Timestep embedding
        + State conditioning
        ↓
        DiT Blocks (4 layers)
        ↓
        Action Logits [B, 49, 7]
    """
    def __init__(
        self,
        num_actions=7,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        num_tokens=49,  # 7x7 grid
        max_seq_len=64,
        dropout=0.1,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.max_seq_len = max_seq_len
        
        # Encoders (already handle CNN tokenization)
        self.state_encoder = StateEncoder(
            grid_size=7,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
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
        
        # State conditioning: project state tokens to conditioning vector
        # StateEncoder outputs [B, 49, 128], we need [B, 128] for conditioning
        self.state_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # [B, 49, 128] -> [B, 1, 128]
            nn.Flatten(start_dim=1),  # [B, 128]
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
        
        Args:
            noisy_actions: [B, num_tokens, hidden_dim] continuous hidden action states
            state: dict with 'grid' [B, 7, 7, 3] and 'direction' [B]
            t: [B] diffusion timestep in [0, 1]
            mask: [B, num_tokens] binary mask (1 = masked position), optional
        
        Returns:
            [B, num_tokens, num_actions] denoised action logits
        """
        B = noisy_actions.shape[0]
        
        # Encode state to tokens
        state_tokens = self.state_encoder(state)  # [B, num_tokens, hidden_dim]
        
        # Project state tokens to conditioning vector
        state_cond = self.state_proj(state_tokens.transpose(1, 2)).transpose(0, 1)  # [B, hidden_dim]
        # Alternative: use mean pooling
        state_cond = state_tokens.mean(dim=1)  # [B, hidden_dim]
        
        # Encode timestep
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        
        # Combine state and time conditioning
        cond_emb = state_cond + t_emb  # [B, hidden_dim]
        
        # Input is already tokenized (noisy_actions are hidden representations)
        x = noisy_actions  # [B, num_tokens, hidden_dim]
        
        # Apply DiT blocks with conditioning
        for block in self.blocks:
            x = block(x, cond_emb)
        
        # Output head
        x = self.final_norm(x)
        logits = self.action_head(x)  # [B, num_tokens, num_actions]
        
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
            hidden_actions: [B, num_tokens, hidden_dim]
            state: state dict
            t: [B] current noise level in [0, 1]
            guidance_scale: float, classifier-free guidance strength
        
        Returns:
            [B, num_tokens, hidden_dim] less noisy hidden actions
        """
        # Forward pass to get action logits
        logits = self.forward(hidden_actions, state, t)  # [B, num_tokens, num_actions]
        
        # Get predicted clean actions
        pred_actions = logits.argmax(dim=-1)  # [B, num_tokens]
        
        # Encode to hidden space
        clean_hidden = self.action_encoder(pred_actions)  # [B, num_tokens, hidden_dim]
        
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
