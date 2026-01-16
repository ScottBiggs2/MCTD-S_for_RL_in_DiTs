"""
Timestep embeddings for diffusion models.
"""
import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding for diffusion.
    
    Maps diffusion timestep t ∈ [0, 1] to embedding space.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        Args:
            t: [batch] timestep values in [0, 1]
        
        Returns:
            [batch, dim] embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Create frequency scales
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        
        # Apply timestep: sin/cos of t * frequencies
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Pad if dim is odd
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device)], dim=-1)
        
        return emb
