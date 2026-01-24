"""
Diffusion Policy for action sequences.

Masked Diffusion Language Model (MDLM) operating on discrete action IDs with
MASK tokens, following the architecture described in `mdlm-fix-guide.md`.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .state_cnn_encoder import StateCNNEncoder
from .action_encoder import ActionEncoder
from .components.dit_block import DiTBlock
from .components.position_embedding import PositionalEmbedding


class DiffusionPolicy(nn.Module):
    """
    Masked Diffusion Language Model for action sequences.
    
    MDLM paradigm:
    - Input: action sequence with some positions replaced by [MASK] token ID
    - Output: logits over original actions at all positions
    - Training: cross-entropy loss on MASKED positions only
    - Inference: iterative unmasking from all-MASK to fully specified sequence
    """
    def __init__(
        self,
        num_actions=7,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        max_seq_len=64,
        dropout=0.1,
        grid_size=19,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.mask_token_id = num_actions  # [MASK] token index
        
        self.grid_size = grid_size
        
        # State encoder: CNN that outputs single embedding vector [B, hidden_dim]
        self.state_encoder = StateCNNEncoder(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_channels=3,
        )
        
        # Action encoder with MASK token support
        self.action_encoder = ActionEncoder(num_actions, hidden_dim)
        
        # Optional mask ratio embedding (scalar ∈ [0,1] → [B, hidden_dim])
        self.mask_ratio_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Positional embeddings for action tokens
        self.action_pos_embed = PositionalEmbedding(
            num_tokens=max_seq_len,
            hidden_dim=hidden_dim
        )
        
        # DiT transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head: predict logits over REAL actions only (exclude MASK)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)
    
    def forward(
        self,
        masked_actions: torch.Tensor,
        state: Dict[str, torch.Tensor],
        mask_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict original actions from a masked action sequence.
        
        Args:
            masked_actions: [B, max_seq_len] discrete action IDs with some positions
                equal to `mask_token_id` (MASK).
            state: dict with 'grid' [B, H, W, 3] and 'direction' [B]
            mask_ratio: [B] fraction of positions masked (optional; used for conditioning)
        
        Returns:
            logits: [B, max_seq_len, num_actions] over REAL actions (0..num_actions-1)
        """
        B, seq_len = masked_actions.shape
        
        # Encode state to single vector
        state_cond = self.state_encoder(state)  # [B, hidden_dim]

        # Encode masked actions (includes MASK embeddings)
        x = self.action_encoder(masked_actions)  # [B, seq_len, hidden_dim]

        # Positional embeddings over action positions
        x = self.action_pos_embed(x)  # [B, seq_len, hidden_dim]

        # Strong state conditioning: add to every token
        x = x + state_cond.unsqueeze(1)  # [B, seq_len, hidden_dim]

        # Optional mask-ratio conditioning using AdaLN cond vector
        if mask_ratio is not None:
            ratio_emb = self.mask_ratio_embed(mask_ratio.unsqueeze(-1))  # [B, hidden_dim]
            cond_emb = ratio_emb
        else:
            cond_emb = torch.zeros(B, self.hidden_dim, device=x.device)

        # Apply DiT blocks with conditioning
        for block in self.blocks:
            x = block(x, cond_emb)
        
        # Output head
        x = self.final_norm(x)
        logits = self.action_head(x)  # [B, seq_len, num_actions]
        
        return logits
    
    @torch.no_grad()
    def sample(
        self,
        state: Dict[str, torch.Tensor],
        seq_len: int,
        num_steps: int = 10,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate an action sequence via iterative unmasking.

        Args:
            state: conditioning state dict
            seq_len: desired sequence length (≤ max_seq_len)
            num_steps: number of unmasking iterations
            temperature: sampling temperature for softmax

        Returns:
            actions: [B, seq_len] generated discrete actions
        """
        B = state["grid"].shape[0]
        device = state["grid"].device

        seq_len = min(seq_len, self.max_seq_len)

        # Start from all-MASK sequence
        actions = torch.full(
            (B, seq_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        is_masked = torch.ones(B, seq_len, dtype=torch.bool, device=device)

        for step in range(num_steps):
            # Decreasing mask ratio over steps
            mask_ratio_val = 1.0 - float(step + 1) / num_steps
            mask_ratio = torch.full((B,), mask_ratio_val, device=device)

            logits = self.forward(actions, state, mask_ratio)  # [B, seq_len, num_actions]
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)

            confidence, predicted = probs.max(dim=-1)  # [B, seq_len]

            # Only consider still-masked positions
            confidence = confidence.masked_fill(~is_masked, -float("inf"))

            num_masked = is_masked.sum(dim=1)  # [B]
            num_to_unmask = torch.ceil(
                num_masked / max(num_steps - step, 1)
            ).long()

            for b in range(B):
                if num_to_unmask[b] <= 0 or num_masked[b] == 0:
                    continue
                masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                if masked_indices.numel() == 0:
                    continue
                conf_at_masked = confidence[b, masked_indices]
                k = min(num_to_unmask[b].item(), masked_indices.numel())
                _, top_idx = conf_at_masked.topk(k)
                unmask_pos = masked_indices[top_idx]

                actions[b, unmask_pos] = predicted[b, unmask_pos]
                is_masked[b, unmask_pos] = False

        # Final clean-up: any remaining MASKs get a final prediction
        if is_masked.any():
            mask_ratio = torch.zeros(B, device=device)
            logits = self.forward(actions, state, mask_ratio)
            final_pred = logits.argmax(dim=-1)
            actions = torch.where(is_masked, final_pred, actions)

        return actions
