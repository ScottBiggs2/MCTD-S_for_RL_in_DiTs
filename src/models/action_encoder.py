"""
Action encoder and decoder using CNN tokenizer.

Architecture: 
- Encoding: Discrete actions -> continuous embeddings (for training)
- Decoding: DiT tokens -> CNN Tokenizer -> Action logits
"""
import torch
import torch.nn as nn
from .components.cnn_tokenizer import ActionCNNTokenizer


class ActionEncoder(nn.Module):
    """
    Encode discrete actions as continuous hidden states.
    
    This is used during training to convert discrete action sequences
    to continuous embeddings for the diffusion process.
    
    For inference, we use ActionCNNTokenizer to decode DiT output tokens.
    """
    def __init__(self, num_actions=7, hidden_dim=128):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Learnable embeddings for each action
        self.action_embedding = nn.Embedding(num_actions, hidden_dim)
        
        # Optional projection layer
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, actions):
        """
        Args:
            actions: [batch, seq_len] discrete action indices
            
        Returns:
            [batch, seq_len, hidden_dim] continuous embeddings
        """
        embeds = self.action_embedding(actions)
        return self.proj(embeds)
    
    def decode(self, hidden_states):
        """
        Decode continuous hidden states back to action logits.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            
        Returns:
            [batch, seq_len, num_actions] logits
        """
        # Compute similarity to each action embedding
        action_embeds = self.action_embedding.weight.t()  # [hidden_dim, num_actions]
        logits = torch.matmul(hidden_states, action_embeds)
        return logits


class ActionDecoder(nn.Module):
    """
    Decode DiT output tokens to action logits using CNN tokenizer.
    
    This is the main decoder used during inference:
    DiT tokens -> CNN Tokenizer -> Action logits
    """
    def __init__(
        self,
        hidden_dim=128,
        num_tokens=49,  # Should match StateEncoder num_tokens
        num_actions=7,
        grid_size=7,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_actions = num_actions
        
        # CNN tokenizer to decode tokens to actions
        self.action_tokenizer = ActionCNNTokenizer(
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            num_actions=num_actions,
            grid_size=grid_size,
        )
    
    def forward(self, tokens):
        """
        Decode tokens to action logits.
        
        Args:
            tokens: (batch, num_tokens, hidden_dim) from DiT
        
        Returns:
            action_logits: (batch, num_tokens, num_actions)
                Note: We get one action per token. For action sequences,
                we may need to aggregate or select specific tokens.
        """
        return self.action_tokenizer(tokens)
    
    def forward_to_sequence(self, tokens, seq_len=None):
        """
        Decode tokens to action sequence.
        
        Args:
            tokens: (batch, num_tokens, hidden_dim)
            seq_len: Desired sequence length (if None, uses num_tokens)
        
        Returns:
            action_logits: (batch, seq_len, num_actions)
        """
        # Get per-token actions
        per_token_logits = self.forward(tokens)  # (batch, num_tokens, num_actions)
        
        if seq_len is None:
            seq_len = self.num_tokens
        
        # For now, we'll use the first seq_len tokens
        # TODO: Could use attention or pooling to aggregate tokens
        if seq_len <= self.num_tokens:
            return per_token_logits[:, :seq_len, :]
        else:
            # Pad or repeat if needed
            padding = seq_len - self.num_tokens
            return torch.cat([
                per_token_logits,
                per_token_logits[:, -padding:, :]  # Repeat last tokens
            ], dim=1)
