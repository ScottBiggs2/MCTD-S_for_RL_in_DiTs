# MDLM Architecture Fix Guide

## Executive Summary

The current implementation is a **confused hybrid** between continuous diffusion (DDPM) and masked diffusion (MDLM). These paradigms are fundamentally incompatible. This document outlines the issues and provides a clear path to a working MDLM implementation for discrete action prediction.

---

## Table of Contents

1. [Critical Issues Identified](#critical-issues-identified)
2. [MDLM Fundamentals](#mdlm-fundamentals)
3. [Files to Modify](#files-to-modify)
4. [Implementation Guide](#implementation-guide)
5. [Training Loop](#training-loop)
6. [Inference/Sampling](#inferencesampling)
7. [Sanity Checks](#sanity-checks)
8. [References](#references)

---

## Critical Issues Identified

### Issue #1: Paradigm Confusion (CRITICAL)

**Location:** `src/models/diffusion_policy.py`, `src/training/mdlm_trainer.py`

**Problem:** The codebase mixes two incompatible approaches:

| Continuous Diffusion (DDPM) | Masked Diffusion (MDLM) |
|----------------------------|------------------------|
| Add Gaussian noise to continuous representations | Replace tokens with [MASK] token |
| Predict noise ε or clean signal x₀ | Predict original discrete tokens |
| MSE loss in continuous space | Cross-entropy loss on discrete classes |
| Timestep t controls noise level | Timestep t controls mask ratio |
| Denoising: interpolate toward prediction | Denoising: replace MASK with predicted token |

**Current code has:**
- Continuous timestep embedding ✓ (DDPM-style)
- Cross-entropy loss on discrete logits ✓ (MDLM-style)
- `denoise_step()` with continuous interpolation ✗ (DDPM-style, but broken)
- No MASK token ✗ (Required for MDLM)
- **Gaussian noise injection in trainer** ✗ (DDPM-style, WRONG for MDLM)

**Fix:** Commit fully to MDLM paradigm for discrete actions.

### Issue #1b: Trainer Uses Gaussian Noise Instead of MASK Token (CRITICAL)

**Location:** `src/training/mdlm_trainer.py` → `train_step()` lines ~147-152

**The Bug:**
```python
# CURRENT (WRONG - this is continuous diffusion, not MDLM!)
noise = torch.randn_like(clean_hidden)  # ❌ Gaussian noise
noisy_hidden = torch.where(
    mask[..., None],
    noise,  # ❌ Replacing with random noise
    clean_hidden
)
```

**Why This Fails:**
- Gaussian noise is continuous and unbounded
- The model sees random vectors at masked positions
- But it's trained with cross-entropy loss on discrete classes
- The model cannot learn a meaningful mapping from "random noise" → "discrete action"
- This is why training doesn't converge!

**The Fix:**
```python
# CORRECT (MDLM - use discrete MASK token)
masked_actions = actions.clone()
masked_actions[mask] = self.model.mask_token_id  # Replace with MASK token (e.g., 7)

# Then encode the masked sequence (including MASK embeddings)
masked_hidden = self.model.action_encoder(masked_actions)  # MASK has its own learned embedding
```

**Key Insight:** In MDLM, the model learns what [MASK] "means" through its embedding. The MASK embedding is a learnable vector that the model uses as a query: "what token should go here?" Random Gaussian noise has no such semantic meaning.

---

### Issue #1c: Input/Output Type Mismatch

**Location:** `src/training/mdlm_trainer.py` → `train_step()`

**Problem:** The trainer operates on continuous hidden states, but MDLM should operate on discrete token IDs:

```python
# CURRENT (WRONG flow):
actions → action_encoder → clean_hidden (continuous)
                                ↓
                    add Gaussian noise at masked positions
                                ↓
                    noisy_hidden (continuous) → model → logits

# CORRECT (MDLM flow):
actions → replace some with MASK token ID → masked_actions (discrete IDs)
                                                    ↓
                                        action_encoder (embeds including MASK)
                                                    ↓
                                        masked_hidden → model → logits
```

**The model's forward() expects:** `noisy_actions: [B, seq_len, hidden_dim]` (continuous)
**MDLM forward() should expect:** `masked_action_ids: [B, seq_len]` (discrete with MASK=7)

---

### Issue #2: Missing MASK Token (CRITICAL)

**Location:** `src/models/action_encoder.py`

**Problem:** For masked diffusion, you need a special `[MASK]` embedding that the model learns to "unmask" into real tokens.

```python
# CURRENT (WRONG):
self.action_embedding = nn.Embedding(num_actions, hidden_dim)  # Only 7 actions (0-6)

# REQUIRED:
self.action_embedding = nn.Embedding(num_actions + 1, hidden_dim)  # 8 embeddings (0-7)
self.mask_token_id = num_actions  # Index 7 = [MASK]
```

Without MASK, what's being fed to the model at masked positions? If it's noise or zeros, that's not MDLM.

---

### Issue #3: Non-Differentiable Inference Path

**Location:** `src/models/diffusion_policy.py` → `denoise_step()`

**Problem:** 
```python
pred_actions = logits.argmax(dim=-1)  # ❌ argmax has NO gradient
clean_hidden = self.action_encoder(pred_actions)  # Can't backprop through this
```

This breaks any training that goes through `denoise_step()`. It's only usable for inference, and the inference logic is wrong for MDLM anyway.

**Fix:** Remove this method or rewrite for proper MDLM sampling.

---

### Issue #4: Weak State Conditioning

**Location:** `src/models/diffusion_policy.py` → `forward()`

**Problem:** State is pooled to a single vector and only added to timestep embedding:
```python
cond_emb = state_cond + t_emb  # [B, hidden_dim] - single vector for everything
```

This single vector must carry ALL information about walls, goals, agent position. It's too weak.

**Fix:** Add state conditioning directly to each action token:
```python
x = x + state_cond.unsqueeze(1)  # Add state to EACH token position
```

---

### Issue #5: Timestep Usage Unclear

**Location:** `src/models/components/timestep_embed.py`, `src/models/diffusion_policy.py`

**Problem:** In MDLM, timestep typically encodes the **mask ratio**, not a noise level. The model should know "what fraction is masked" to calibrate confidence.

**Options:**
1. Remove timestep entirely (simplest BERT-style approach)
2. Use timestep to encode mask ratio (more flexible, allows curriculum)
3. Keep timestep but interpret as mask ratio, not noise level

---

## MDLM Fundamentals

### What is Masked Diffusion?

Masked Diffusion Language Models (MDLM) are discrete diffusion models that:

1. **Forward process:** Progressively mask tokens (replace with [MASK])
2. **Reverse process:** Predict original tokens at masked positions
3. **Training:** Random masking with cross-entropy loss
4. **Inference:** Iteratively unmask, starting from all-masked

### Key Equations

**Mask Schedule (Cosine):**
```
mask_ratio(t) = cos(π/2 × (1 - t))

t=0 → mask_ratio=1.0 (all masked)
t=1 → mask_ratio=0.0 (none masked)
```

**Training Objective:**
```
L = -𝔼[log p(x_masked | x_unmasked, state)]

Only compute loss on MASKED positions (like BERT MLM)
```

**Inference (Iterative Unmasking):**
```
Start: all tokens = [MASK]
For t in [0, 0.1, 0.2, ..., 1.0]:
    1. Forward pass → get logits for all positions
    2. Sample/argmax at currently masked positions
    3. Unmask most confident predictions (keep some masked)
    4. Repeat until t=1 (all unmasked)
```

---

## Trainer-Specific Issues (`mdlm_trainer.py`)

### Current Trainer Analysis

The trainer has several good elements that should be preserved:
- ✅ Cosine mask schedule implementation
- ✅ Curriculum learning for mask ratio (nice feature!)
- ✅ Loss computed only on masked positions
- ✅ Gradient clipping
- ✅ Learning rate scheduling

**But the core training loop is fundamentally broken:**

### Problem 1: Gaussian Noise Instead of MASK Token

```python
# Line ~140-152 in train_step() - THIS IS THE CORE BUG
clean_hidden = self.model.action_encoder(actions)  # Encode to continuous space

# ... then ...

noise = torch.randn_like(clean_hidden)  # ❌ Generate random noise
noisy_hidden = torch.where(
    mask[..., None],
    noise,  # ❌ Replace masked positions with noise
    clean_hidden
)
```

**This is continuous diffusion applied to a discrete problem.** The model receives random Gaussian vectors at masked positions but must output discrete class probabilities. There's no learnable relationship between "random noise" and "correct action."

### Problem 2: Forward Pass Signature

```python
# Current call:
logits = self.model(noisy_hidden, states, t, mask)
# model.forward expects: noisy_actions: [B, seq_len, hidden_dim]

# Should be:
logits = self.model(masked_action_ids, states, mask_ratio)
# model.forward should expect: masked_actions: [B, seq_len] (discrete IDs)
```

### Problem 3: Evaluation Uses Same Broken Pattern

The `evaluate()` method has the same issues - it uses Gaussian noise instead of MASK tokens.

---

## Files to Modify

### Priority 1: Core Architecture

| File | Changes Needed |
|------|----------------|
| `src/models/action_encoder.py` | Add MASK token, update embedding size |
| `src/models/diffusion_policy.py` | Fix forward(), remove/fix denoise_step(), strengthen state conditioning |
| `src/training/mdlm_trainer.py` | Implement proper MDLM training loop |

### Priority 2: Supporting Changes

| File | Changes Needed |
|------|----------------|
| `src/models/components/timestep_embed.py` | Optional: repurpose for mask_ratio encoding |
| `src/models/components/dit_block.py` | Optional: simplify if not using timestep |
| `scripts/train_baseline.py` | Update to use new training loop |

---

## Implementation Guide

### Step 1: Fix ActionEncoder (`src/models/action_encoder.py`)

```python
class ActionEncoder(nn.Module):
    """
    Encode discrete actions as continuous hidden states.
    Includes MASK token for masked diffusion.
    """
    def __init__(self, num_actions=7, hidden_dim=128):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # CRITICAL: Add +1 for MASK token
        self.mask_token_id = num_actions  # Index 7 = [MASK]
        self.vocab_size = num_actions + 1  # 8 total embeddings
        
        # Learnable embeddings: 7 actions + 1 MASK
        self.action_embedding = nn.Embedding(self.vocab_size, hidden_dim)
        
        # Optional projection layer
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, actions):
        """
        Args:
            actions: [B, seq_len] discrete action indices (0-6 for actions, 7 for MASK)
        Returns:
            [B, seq_len, hidden_dim] continuous embeddings
        """
        embeds = self.action_embedding(actions)
        return self.proj(embeds)
    
    def get_mask_embedding(self, batch_size, seq_len, device):
        """Get embedding for all-MASK sequence (for inference start)."""
        mask_ids = torch.full((batch_size, seq_len), self.mask_token_id, 
                              dtype=torch.long, device=device)
        return self.forward(mask_ids)
```

---

### Step 2: Fix DiffusionPolicy (`src/models/diffusion_policy.py`)

```python
class DiffusionPolicy(nn.Module):
    """
    Masked Diffusion Language Model for action sequences.
    
    MDLM Paradigm:
    - Input: Action sequence with some positions replaced by [MASK]
    - Output: Logits predicting original actions at ALL positions
    - Training: Cross-entropy loss ONLY on masked positions
    - Inference: Iteratively unmask from all-MASK to all-predicted
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
        self.mask_token_id = num_actions  # [MASK] = 7
        
        # State encoder (outputs single vector)
        self.state_encoder = StateCNNEncoder(
            grid_size=grid_size,
            hidden_dim=hidden_dim,
            num_channels=3,
        )
        
        # Action encoder WITH mask token
        self.action_encoder = ActionEncoder(num_actions, hidden_dim)
        
        # Mask ratio embedding (optional but helpful)
        # Tells model what fraction is masked - helps calibrate confidence
        self.mask_ratio_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Positional embeddings for action sequence
        self.action_pos_embed = PositionalEmbedding(
            num_tokens=max_seq_len,
            hidden_dim=hidden_dim
        )
        
        # DiT transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head - predicts action logits (NOT including MASK in output)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions)  # 7 actions, not 8
    
    def forward(
        self,
        masked_actions: torch.Tensor,  # [B, seq_len] with MASK tokens
        state: Dict[str, torch.Tensor],
        mask_ratio: torch.Tensor = None,  # [B] fraction masked, optional
    ) -> torch.Tensor:
        """
        Predict original actions from masked sequence.
        
        Args:
            masked_actions: [B, seq_len] action indices with some = mask_token_id (7)
            state: dict with 'grid' and 'direction'
            mask_ratio: [B] fraction of tokens masked (optional, for conditioning)
        
        Returns:
            logits: [B, seq_len, num_actions] prediction logits for all positions
        """
        B, seq_len = masked_actions.shape
        
        # Encode state to single vector
        state_cond = self.state_encoder(state)  # [B, hidden_dim]
        
        # Encode masked action sequence
        x = self.action_encoder(masked_actions)  # [B, seq_len, hidden_dim]
        
        # Add positional embeddings
        x = self.action_pos_embed(x)  # [B, seq_len, hidden_dim]
        
        # STRONGER STATE CONDITIONING: Add state to each token
        # This is critical - model needs state info at every position
        x = x + state_cond.unsqueeze(1)  # [B, seq_len, hidden_dim]
        
        # Optional: encode mask ratio for conditioning
        if mask_ratio is not None:
            ratio_emb = self.mask_ratio_embed(mask_ratio.unsqueeze(-1))  # [B, hidden_dim]
            cond = ratio_emb
        else:
            cond = torch.zeros(B, self.hidden_dim, device=x.device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cond)  # cond used for AdaLN
        
        # Output logits
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
        Generate action sequence via iterative unmasking.
        
        Args:
            state: State dict for conditioning
            seq_len: Length of sequence to generate
            num_steps: Number of unmasking steps
            temperature: Sampling temperature (1.0 = normal, <1 = greedy)
        
        Returns:
            actions: [B, seq_len] generated action sequence
        """
        B = state['grid'].shape[0]
        device = state['grid'].device
        
        # Start with all MASK tokens
        actions = torch.full((B, seq_len), self.mask_token_id, 
                            dtype=torch.long, device=device)
        
        # Track which positions are still masked
        is_masked = torch.ones(B, seq_len, dtype=torch.bool, device=device)
        
        # Iteratively unmask
        for step in range(num_steps):
            # Current mask ratio (decreases each step)
            mask_ratio = 1.0 - (step + 1) / num_steps
            mask_ratio_t = torch.full((B,), mask_ratio, device=device)
            
            # Forward pass
            logits = self.forward(actions, state, mask_ratio_t)  # [B, seq_len, num_actions]
            
            # Get probabilities
            probs = F.softmax(logits / temperature, dim=-1)  # [B, seq_len, num_actions]
            
            # Get confidence (max probability) at each position
            confidence, predicted = probs.max(dim=-1)  # [B, seq_len]
            
            # Only consider masked positions
            confidence = confidence.masked_fill(~is_masked, -float('inf'))
            
            # Determine how many to unmask this step
            num_masked = is_masked.sum(dim=1)  # [B]
            num_to_unmask = torch.ceil(num_masked / (num_steps - step)).long()
            
            # Unmask most confident predictions
            for b in range(B):
                if num_to_unmask[b] > 0:
                    # Get indices of masked positions sorted by confidence
                    masked_indices = is_masked[b].nonzero(as_tuple=True)[0]
                    if len(masked_indices) > 0:
                        conf_at_masked = confidence[b, masked_indices]
                        _, top_indices = conf_at_masked.topk(
                            min(num_to_unmask[b].item(), len(masked_indices))
                        )
                        unmask_positions = masked_indices[top_indices]
                        
                        # Unmask: replace MASK with predicted action
                        actions[b, unmask_positions] = predicted[b, unmask_positions]
                        is_masked[b, unmask_positions] = False
        
        # Final pass: fill any remaining masked positions
        if is_masked.any():
            logits = self.forward(actions, state, torch.zeros(B, device=device))
            predicted = logits.argmax(dim=-1)
            actions = torch.where(is_masked, predicted, actions)
        
        return actions
```

---

### Step 3: Fix Training Loop (`src/training/mdlm_trainer.py`)

**Keep:** Curriculum learning, cosine schedule, LR scheduling, gradient clipping
**Fix:** Replace Gaussian noise with MASK token, change input from continuous to discrete

```python
"""
Masked Diffusion Language Model (MDLM) Trainer - FIXED VERSION

Key changes from original:
1. Use MASK token instead of Gaussian noise
2. Input discrete action IDs (with MASK), not continuous hidden states
3. Model handles embedding internally
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
import math


class MaskedDiffusionTrainer:
    """
    Trainer for Masked Diffusion Language Model (MDLM).
    
    FIXED: Uses discrete MASK tokens instead of Gaussian noise.
    """
    def __init__(
        self,
        model,  # DiffusionPolicy with MASK token support
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cpu',
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # CRITICAL: Get MASK token ID from model
        self.mask_token_id = model.mask_token_id  # Should be 7 (num_actions)
        
        # Optimizer (unchanged)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
        )
        
        # LR scheduler (unchanged)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5,
            patience=config.get('lr_patience', 5),
        )
        
        # Loss function (unchanged)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Curriculum learning settings (keep these - they're good!)
        self.use_curriculum = config.get('use_mask_curriculum', True)
        self.min_mask_ratio = config.get('min_mask_ratio', 0.1)
        self.max_mask_ratio = config.get('max_mask_ratio', 0.75)
        self.curriculum_warmup_epochs = config.get('mask_curriculum_warmup_epochs', None)
        self.current_epoch = 0
    
    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        """Get mask ratio from timestep. (Keep existing implementation)"""
        # ... (keep your existing curriculum logic - it's good!)
        schedule = self.config.get('mask_schedule', 'cosine')
        if schedule == 'cosine':
            mask_ratio = torch.cos(math.pi / 2 * (1 - t))
        elif schedule == 'linear':
            mask_ratio = t
        else:
            mask_ratio = torch.full_like(t, 0.5)
        
        # Apply curriculum if enabled
        if self.use_curriculum:
            target = self.get_curriculum_mask_ratio()
            # Blend toward target
            mask_ratio = mask_ratio * 0.3 + target * 0.7
        
        return torch.clamp(mask_ratio, 0.05, 0.95)
    
    def create_masked_input(
        self,
        actions: torch.Tensor,  # [B, seq_len] ground truth action IDs
        lengths: torch.Tensor,  # [B] actual sequence lengths
    ):
        """
        Create masked input using MASK token (NOT Gaussian noise!).
        
        Returns:
            masked_actions: [B, seq_len] with some positions = mask_token_id
            mask: [B, seq_len] boolean, True = masked position
            mask_ratio: [B] actual mask ratio per sample
        """
        B, seq_len = actions.shape
        device = actions.device
        
        # Sample timestep for each sample
        t = torch.rand(B, device=device)
        target_mask_ratio = self.get_mask_ratio(t)  # [B]
        
        # Create random mask
        rand = torch.rand(B, seq_len, device=device)
        mask = rand < target_mask_ratio.unsqueeze(1)  # [B, seq_len]
        
        # DON'T mask padding positions
        position_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_mask = position_indices < lengths.unsqueeze(1)  # [B, seq_len]
        mask = mask & valid_mask
        
        # Ensure at least one position is masked (for learning signal)
        for b in range(B):
            if not mask[b].any() and lengths[b] > 0:
                # Mask a random valid position
                valid_positions = valid_mask[b].nonzero(as_tuple=True)[0]
                if len(valid_positions) > 0:
                    random_idx = valid_positions[torch.randint(len(valid_positions), (1,))]
                    mask[b, random_idx] = True
        
        # CRITICAL FIX: Replace masked positions with MASK token ID
        # NOT Gaussian noise!
        masked_actions = actions.clone()
        masked_actions[mask] = self.mask_token_id  # ← THIS IS THE KEY FIX
        
        actual_mask_ratio = mask.float().sum(dim=1) / lengths.float().clamp(min=1)
        
        return masked_actions, mask, actual_mask_ratio
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step - FIXED VERSION.
        
        Key change: Use MASK tokens, not Gaussian noise.
        """
        self.model.train()
        
        # Unpack batch
        states = {k: v.to(self.device) for k, v in batch['states'].items()}
        actions = batch['actions'].to(self.device)  # [B, seq_len] ground truth
        lengths = batch['length'].to(self.device)   # [B] actual lengths
        
        B, seq_len = actions.shape
        max_seq_len = self.model.max_seq_len
        
        # Pad/truncate actions to max_seq_len
        if seq_len < max_seq_len:
            # Pad with a padding action (use action 0 or last action)
            pad_value = 0  # Or actions[:, -1:].expand(B, max_seq_len - seq_len)
            padding = torch.full((B, max_seq_len - seq_len), pad_value, 
                                dtype=torch.long, device=self.device)
            actions_padded = torch.cat([actions, padding], dim=1)
            # Don't update lengths - they track actual content
        elif seq_len > max_seq_len:
            actions_padded = actions[:, :max_seq_len]
            lengths = lengths.clamp(max=max_seq_len)
        else:
            actions_padded = actions
        
        # FIXED: Create masked input with MASK tokens (not noise!)
        masked_actions, mask, mask_ratio = self.create_masked_input(
            actions_padded, lengths
        )
        
        # FIXED: Forward pass with discrete masked action IDs
        # Model will embed them internally (including MASK token)
        logits = self.model(masked_actions, states, mask_ratio)  # [B, max_seq_len, num_actions]
        
        # Compute loss ONLY on masked positions (this was correct!)
        loss_per_token = self.criterion(
            logits.reshape(-1, self.model.num_actions),
            actions_padded.reshape(-1)
        ).reshape(B, max_seq_len)
        
        # Loss only on masked positions
        masked_loss = loss_per_token * mask.float()
        loss = masked_loss.sum() / mask.sum().clamp(min=1)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Metrics
        with torch.no_grad():
            pred_actions = logits.argmax(dim=-1)
            correct = (pred_actions == actions_padded) & mask
            accuracy = correct.sum().float() / mask.sum().clamp(min=1)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'mask_ratio': mask_ratio.mean().item(),
            'num_masked': mask.sum().item(),
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set - FIXED VERSION."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            states = {k: v.to(self.device) for k, v in batch['states'].items()}
            actions = batch['actions'].to(self.device)
            lengths = batch['length'].to(self.device)
            
            B, seq_len = actions.shape
            max_seq_len = self.model.max_seq_len
            
            # Pad/truncate
            if seq_len < max_seq_len:
                padding = torch.zeros(B, max_seq_len - seq_len, 
                                     dtype=torch.long, device=self.device)
                actions_padded = torch.cat([actions, padding], dim=1)
            elif seq_len > max_seq_len:
                actions_padded = actions[:, :max_seq_len]
                lengths = lengths.clamp(max=max_seq_len)
            else:
                actions_padded = actions
            
            # Fixed mask ratio for validation (50%)
            mask_ratio = torch.full((B,), 0.5, device=self.device)
            rand = torch.rand(B, max_seq_len, device=self.device)
            mask = rand < 0.5
            
            # Don't mask padding
            position_indices = torch.arange(max_seq_len, device=self.device).unsqueeze(0)
            valid_mask = position_indices < lengths.unsqueeze(1)
            mask = mask & valid_mask
            
            # FIXED: Use MASK token, not noise
            masked_actions = actions_padded.clone()
            masked_actions[mask] = self.mask_token_id
            
            # Forward
            logits = self.model(masked_actions, states, mask_ratio)
            
            # Loss on masked positions
            loss_per_token = self.criterion(
                logits.reshape(-1, self.model.num_actions),
                actions_padded.reshape(-1)
            ).reshape(B, max_seq_len)
            
            masked_loss = (loss_per_token * mask.float()).sum()
            total_loss += masked_loss.item()
            
            # Accuracy
            pred_actions = logits.argmax(dim=-1)
            correct = (pred_actions == actions_padded) & mask
            total_correct += correct.sum().item()
            total_masked += mask.sum().item()
        
        return {
            'loss': total_loss / max(total_masked, 1),
            'accuracy': total_correct / max(total_masked, 1),
        }
    
    # Keep the rest of the methods (train_epoch, train, etc.) - they're fine
```

**Summary of Trainer Changes:**

| Original | Fixed |
|----------|-------|
| `noise = torch.randn_like(clean_hidden)` | `masked_actions[mask] = self.mask_token_id` |
| `noisy_hidden = torch.where(mask, noise, clean_hidden)` | `masked_actions = actions.clone(); masked_actions[mask] = MASK` |
| `model(noisy_hidden, states, t, mask)` | `model(masked_actions, states, mask_ratio)` |
| Input: continuous hidden states | Input: discrete action IDs with MASK tokens |

---

## Inference/Sampling

### Iterative Unmasking Algorithm

```
Input: state, desired sequence length L, num_steps T
Output: action sequence of length L

1. Initialize: actions = [MASK, MASK, ..., MASK]  (L tokens)
2. For t = 1 to T:
   a. mask_ratio = 1 - t/T
   b. logits = model(actions, state, mask_ratio)
   c. confidence = softmax(logits).max(dim=-1)
   d. For each still-masked position:
      - If confidence > threshold OR this is last step:
        - Unmask: replace MASK with argmax(logits)
3. Return actions
```

### Sampling Strategies

1. **Greedy:** Always pick argmax (temperature=0)
2. **Sampling:** Sample from softmax (temperature=1)
3. **Top-k:** Sample from top k most likely
4. **Nucleus (top-p):** Sample from smallest set with cumulative prob > p

---

## Sanity Checks

### Test 1: Overfit Single Example

```python
# Train on ONE trajectory for 1000 steps
# Loss should approach 0, accuracy should approach 1.0
# If not: architecture is broken

single_batch = next(iter(train_loader))
for step in range(1000):
    loss, metrics = trainer.train_step(single_batch)
    if step % 100 == 0:
        print(f"Step {step}: loss={loss:.4f}, acc={metrics['accuracy']:.4f}")

# Expected: loss < 0.1, accuracy > 0.95 by step 1000
```

### Test 2: Check Embeddings

```python
# Verify MASK token is distinct from action tokens
with torch.no_grad():
    action_embs = model.action_encoder.action_embedding.weight[:7]  # Action embeddings
    mask_emb = model.action_encoder.action_embedding.weight[7:8]    # MASK embedding
    
    # Compute similarities
    sims = F.cosine_similarity(action_embs, mask_emb, dim=-1)
    print(f"MASK similarity to actions: {sims}")
    # Should be low (< 0.5) - MASK should be distinct
```

### Test 3: Check Position Embeddings

```python
# Model should give different outputs for different positions
with torch.no_grad():
    # All MASK input
    B, L = 1, 32
    masked_input = torch.full((B, L), model.mask_token_id, device=device)
    state = get_dummy_state(B, device)
    
    logits = model(masked_input, state, mask_ratio=torch.ones(B, device=device))
    
    # Check variance across positions
    pos_variance = logits.var(dim=1).mean()
    print(f"Position variance: {pos_variance:.4f}")
    # Should be > 0 - different positions should get different predictions
```

### Test 4: State Conditioning

```python
# Different states should give different outputs
with torch.no_grad():
    masked_input = torch.full((2, 32), model.mask_token_id, device=device)
    
    state1 = get_state_for_maze_1()
    state2 = get_state_for_maze_2()  # Different maze
    
    logits1 = model(masked_input[:1], state1, ...)
    logits2 = model(masked_input[:1], state2, ...)
    
    diff = (logits1 - logits2).abs().mean()
    print(f"Output difference for different states: {diff:.4f}")
    # Should be significant (> 0.1) - model should respond to state
```

---

## References

### Key Papers

1. **MDLM (Masked Diffusion Language Models)**
   - "Simple and Effective Masked Diffusion Language Models" (Sahoo et al., 2024)
   - https://arxiv.org/abs/2406.07524
   - Key insight: Simpler than continuous diffusion for discrete data

2. **BERT (Original Masked LM)**
   - "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
   - Foundation for masked prediction

3. **DiT (Diffusion Transformer)**
   - "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
   - AdaLN architecture you're using

4. **Diffuser (Diffusion for Planning)**
   - "Planning with Diffusion for Flexible Behavior Synthesis" (Janner et al., 2022)
   - Continuous diffusion for trajectory generation

5. **MCTD (Monte Carlo Tree Diffusion)**
   - "Monte Carlo Tree Diffusion for System 2 Planning" (Yoon et al., 2025)
   - Official implementation: https://github.com/ahn-ml/mctd

### Reference Implementations

1. **ZHZisZZ/dLLM** - Diffusion LLM with working Qwen3 checkpoints
   - https://github.com/ZHZisZZ/dLLM

2. **Salesforce CoDA** - Continuous diffusion for discrete actions
   - Different approach: embeds discrete actions in continuous space

3. **Apple DiffuCoder** - Diffusion for code generation
   - Uses MDLM-style masking

---

## Visual: Original vs Fixed Training Flow

### ❌ Original (Broken) Flow

```
Ground Truth Actions: [2, 0, 2, 2, 1, 0]  (move, left, move, move, right, left)
                           ↓
              action_encoder.forward()
                           ↓
Clean Hidden:     [h₂, h₀, h₂, h₂, h₁, h₀]  (continuous vectors)
                           ↓
              Mask positions 1, 3, 5
                           ↓
              Replace with GAUSSIAN NOISE ← ❌ WRONG!
                           ↓
Noisy Hidden:     [h₂, 𝒩(0,1), h₂, 𝒩(0,1), h₁, 𝒩(0,1)]  (noise has no meaning!)
                           ↓
                    model.forward()
                           ↓
Logits:           [l₀, l₁, l₂, l₃, l₄, l₅]  
                           ↓
              CrossEntropy on masked positions
                           ↓
              Model learns: "random noise → action" ← IMPOSSIBLE!
```

### ✅ Fixed (MDLM) Flow

```
Ground Truth Actions: [2, 0, 2, 2, 1, 0]  (move, left, move, move, right, left)
                           ↓
              Mask positions 1, 3, 5
                           ↓
              Replace with MASK TOKEN ID (7) ← ✅ CORRECT!
                           ↓
Masked Actions:   [2, 7, 2, 7, 1, 7]  (discrete IDs with MASK=7)
                           ↓
              action_encoder.forward()  (embeds including learned MASK embedding)
                           ↓
Masked Hidden:    [h₂, h_MASK, h₂, h_MASK, h₁, h_MASK]  (MASK has learned meaning!)
                           ↓
                    model.forward()
                           ↓
Logits:           [l₀, l₁, l₂, l₃, l₄, l₅]  
                           ↓
              CrossEntropy on masked positions
                           ↓
              Model learns: "h_MASK + context → original action" ← LEARNABLE!
```

### Why MASK Token Works

The MASK token embedding is **learned during training**. It becomes a "query vector" that means:
> "Based on the surrounding context (other actions + state), what action should go here?"

Gaussian noise has no such semantic meaning - it's just random, and the model cannot learn any consistent mapping from arbitrary random vectors to correct actions.

---

## Summary Checklist

### ActionEncoder (`src/models/action_encoder.py`)
- [ ] Add MASK token: `self.mask_token_id = num_actions` (index 7)
- [ ] Update embedding size: `nn.Embedding(num_actions + 1, hidden_dim)`
- [ ] Add helper method `get_mask_embedding()`

### DiffusionPolicy (`src/models/diffusion_policy.py`)
- [ ] Add `self.mask_token_id = num_actions` attribute
- [ ] Change `forward()` to accept discrete action IDs, not continuous hidden
- [ ] Model embeds actions internally (including MASK tokens)
- [ ] Add state conditioning to each token: `x = x + state_cond.unsqueeze(1)`
- [ ] Remove or repurpose timestep for mask_ratio encoding
- [ ] Remove broken `denoise_step()` or rewrite for MDLM sampling
- [ ] Implement `sample()` method with iterative unmasking

### MaskedDiffusionTrainer (`src/training/mdlm_trainer.py`)
- [ ] **CRITICAL:** Remove Gaussian noise injection
- [ ] **CRITICAL:** Use MASK token: `masked_actions[mask] = self.mask_token_id`
- [ ] Change input from continuous hidden to discrete action IDs
- [ ] Update `create_masked_input()` to return masked action IDs
- [ ] Update `train_step()` forward call signature
- [ ] Update `evaluate()` with same fixes
- [ ] Keep curriculum learning (it's good!)
- [ ] Keep loss-on-masked-only (it's correct!)

### Sanity Checks (Before Full Training)
- [ ] Run single-example overfit test (loss → 0, acc → 1.0)
- [ ] Verify MASK embedding is distinct from action embeddings
- [ ] Verify position embeddings produce different outputs per position  
- [ ] Verify different states produce different outputs

### After Fixes Work
- [ ] Train on full dataset
- [ ] Evaluate generation quality with `model.sample()`
- [ ] Test in actual maze environment
- [ ] Then add MCTD search layer

---

## Quick Reference: Exact Lines to Change

### `mdlm_trainer.py` - The Critical Fix

**Find this code block (~lines 140-152):**
```python
# REMOVE/REPLACE THIS:
clean_hidden = self.model.action_encoder(actions)
# ...
noise = torch.randn_like(clean_hidden)
noisy_hidden = torch.where(
    mask[..., None],
    noise,
    clean_hidden
)
logits = self.model(noisy_hidden, states, t, mask)
```

**Replace with:**
```python
# FIXED:
masked_actions = actions_padded.clone()
masked_actions[mask] = self.model.mask_token_id  # Use MASK token, not noise!
logits = self.model(masked_actions, states, mask_ratio)  # Model embeds internally
```

### `action_encoder.py` - Add MASK Token

**Find:**
```python
self.action_embedding = nn.Embedding(num_actions, hidden_dim)
```

**Replace with:**
```python
self.mask_token_id = num_actions  # MASK = 7
self.action_embedding = nn.Embedding(num_actions + 1, hidden_dim)  # +1 for MASK
```

### `diffusion_policy.py` - Change Forward Signature

**Find:**
```python
def forward(
    self,
    noisy_actions: torch.Tensor,  # [B, seq_len, hidden_dim] <- WRONG
    ...
```

**Replace with:**
```python
def forward(
    self,
    masked_actions: torch.Tensor,  # [B, seq_len] discrete IDs with MASK tokens <- CORRECT
    ...
```

**Inside forward(), change:**
```python
# REMOVE:
x = noisy_actions  # Already continuous

# REPLACE WITH:
x = self.action_encoder(masked_actions)  # Embed discrete IDs (including MASK)
```

---

## Next Steps After Fixes

1. **Verify single-example overfitting** (sanity check #1)
2. **Train on full dataset** with proper metrics logging
3. **Evaluate generation quality** by sampling and executing in environment
4. **Then** add MCTD search on top of working base model
