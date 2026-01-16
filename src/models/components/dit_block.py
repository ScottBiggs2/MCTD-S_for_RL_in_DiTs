"""
DiT (Diffusion Transformer) block with adaptive layer norm.
"""
import torch
import torch.nn as nn


class DiTBlock(nn.Module):
    """
    Transformer block with adaptive layer norm (DiT style).
    
    Conditions on timestep embedding for adaptive normalization.
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Self-attention
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Adaptive layer norm: generates 6 parameters from timestep embedding
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6)
        )
    
    def forward(self, x, t_emb):
        """
        Args:
            x: [batch, seq_len, hidden_dim] input tokens
            t_emb: [batch, hidden_dim] timestep embedding
        
        Returns:
            [batch, seq_len, hidden_dim] output tokens
        """
        # Generate adaptive LayerNorm parameters from timestep
        ada_params = self.adaLN(t_emb)  # [batch, hidden_dim * 6]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            ada_params.chunk(6, dim=-1)
        
        # Self-attention with adaptive norm
        normed = self.norm1(x)
        # Apply adaptive scale and shift
        normed = normed * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        attn_out, _ = self.attn(normed, normed, normed)
        # Apply gating
        x = x + gate_msa[:, None, :] * attn_out
        
        # MLP with adaptive norm
        normed = self.norm2(x)
        # Apply adaptive scale and shift
        normed = normed * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        mlp_out = self.mlp(normed)
        # Apply gating
        x = x + gate_mlp[:, None, :] * mlp_out
        
        return x
