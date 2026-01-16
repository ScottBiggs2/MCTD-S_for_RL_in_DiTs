"""
Position embeddings for tokenized states.
"""
import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """
    Learned positional embeddings for token sequences.
    
    Used after CNN tokenization to add positional information:
    tokens + pos_embed -> DiT input
    """
    def __init__(self, num_tokens, hidden_dim):
        """
        Args:
            num_tokens: Maximum number of tokens (e.g., 49 for 7x7 grid)
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        
        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)
    
    def forward(self, tokens):
        """
        Add positional embeddings to tokens.
        
        Args:
            tokens: (batch, seq_len, hidden_dim)
        
        Returns:
            tokens_with_pos: (batch, seq_len, hidden_dim)
        """
        seq_len = tokens.shape[1]
        
        # Use first seq_len positions
        pos_emb = self.pos_embed[:, :seq_len, :]
        
        return tokens + pos_emb


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embeddings (non-learned, like in Transformer).
    """
    def __init__(self, num_tokens, hidden_dim):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        
        # Pre-compute sinusoidal embeddings
        position = torch.arange(num_tokens).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           -(math.log(10000.0) / hidden_dim))
        
        pos_embed = torch.zeros(num_tokens, hidden_dim)
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))  # (1, num_tokens, hidden_dim)
    
    def forward(self, tokens):
        """
        Add sinusoidal positional embeddings to tokens.
        
        Args:
            tokens: (batch, seq_len, hidden_dim)
        
        Returns:
            tokens_with_pos: (batch, seq_len, hidden_dim)
        """
        seq_len = tokens.shape[1]
        pos_emb = self.pos_embed[:, :seq_len, :]
        
        return tokens + pos_emb
