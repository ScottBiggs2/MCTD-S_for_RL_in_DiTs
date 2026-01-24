"""
Masked Diffusion Language Model (MDLM) Trainer.

Implements masked diffusion training with cosine masking schedule.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
import math

from ..models.diffusion_policy import DiffusionPolicy


class MaskedDiffusionTrainer:
    """
    Trainer for Masked Diffusion Language Model (MDLM).
    
    Implements cosine masking schedule and masked diffusion loss.
    """
    def __init__(
        self,
        model: DiffusionPolicy,
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
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('lr_patience', 5),
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Diffusion steps
        self.num_diffusion_steps = config.get('num_diffusion_steps', 100)
        
        # Curriculum learning for masking
        self.use_curriculum = config.get('use_mask_curriculum', True)
        self.curriculum_schedule = config.get('mask_curriculum_schedule', 'linear')  # 'linear' or 'cosine'
        self.min_mask_ratio = config.get('min_mask_ratio', 0.1)  # Start with 10% masking
        self.max_mask_ratio = config.get('max_mask_ratio', 0.75)  # End with 75% masking
        self.curriculum_warmup_epochs = config.get('mask_curriculum_warmup_epochs', None)  # None = use num_epochs
        self.current_epoch = 0
    
    def get_curriculum_mask_ratio(self) -> float:
        """
        Get current curriculum mask ratio based on training progress.
        
        Returns:
            Current target mask ratio (between min_mask_ratio and max_mask_ratio)
        """
        if not self.use_curriculum:
            # No curriculum: use average of min and max
            return (self.min_mask_ratio + self.max_mask_ratio) / 2
        
        warmup_epochs = self.curriculum_warmup_epochs
        if warmup_epochs is None:
            # Use num_epochs from config if available, otherwise default to 50
            warmup_epochs = self.config.get('num_epochs', 50)
        
        # Progress from 0 to 1 over warmup_epochs
        progress = min(1.0, self.current_epoch / warmup_epochs)
        
        if self.curriculum_schedule == 'linear':
            # Linear interpolation
            current_ratio = self.min_mask_ratio + (self.max_mask_ratio - self.min_mask_ratio) * progress
        elif self.curriculum_schedule == 'cosine':
            # Cosine interpolation (smooth start, faster end)
            current_ratio = self.min_mask_ratio + (self.max_mask_ratio - self.min_mask_ratio) * (1 - math.cos(progress * math.pi / 2))
        else:
            raise ValueError(f"Unknown curriculum schedule: {self.curriculum_schedule}")
        
        return current_ratio
    
    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get mask ratio based on timestep and curriculum learning.
        
        With curriculum learning:
        - The target mask ratio increases over epochs
        - Timestep t still controls variation within a batch
        - The actual mask ratio is a combination of curriculum target and t
        
        Args:
            t: [batch] timestep values in [0, 1]
        
        Returns:
            [batch] mask ratios in [0, 1]
        """
        schedule = self.config.get('mask_schedule', 'cosine')
        
        if self.use_curriculum:
            # Curriculum learning: get target ratio for current epoch
            target_ratio = self.get_curriculum_mask_ratio()
            
            # Use timestep to add variation around the target
            # t=0 → lower than target, t=1 → higher than target
            # Range: [target - spread, target + spread]
            spread = min(0.2, (self.max_mask_ratio - self.min_mask_ratio) / 2)
            
            if schedule == 'cosine':
                # Cosine variation: t controls how much we deviate from target
                # t=0 → target - spread, t=1 → target + spread
                variation = spread * (2 * t - 1)  # [-spread, +spread]
                mask_ratio = target_ratio + variation
            elif schedule == 'linear':
                # Linear variation
                variation = spread * (2 * t - 1)
                mask_ratio = target_ratio + variation
            else:
                # Constant: just use target
                mask_ratio = torch.full_like(t, target_ratio)
            
            # Clamp to valid range
            mask_ratio = torch.clamp(mask_ratio, 0.0, 1.0)
        else:
            # No curriculum: original behavior
            if schedule == 'cosine':
                # Cosine schedule: more masking at higher t
                mask_ratio = torch.cos(math.pi / 2 * (1 - t))
            elif schedule == 'linear':
                mask_ratio = t
            elif schedule == 'constant':
                mask_ratio = torch.full_like(t, 0.15)
            else:
                raise ValueError(f"Unknown schedule: {schedule}")
        
        return mask_ratio
    
    def create_masked_input(
        self,
        actions: torch.Tensor,
        lengths: torch.Tensor,
    ):
        """
        Create masked input using the model's MASK token (discrete MDLM).

        Args:
            actions: [B, seq_len] ground-truth action IDs
            lengths: [B] actual (unpadded) sequence lengths

        Returns:
            masked_actions: [B, seq_len] with some positions = mask_token_id
            mask: [B, seq_len] boolean tensor, True = masked position
            mask_ratio: [B] actual mask ratio per sample
        """
        B, seq_len = actions.shape
        device = actions.device

        # Sample timestep per sample and map to target mask ratio
        t = torch.rand(B, device=device)
        target_mask_ratio = self.get_mask_ratio(t)  # [B]

        # Initial random mask
        rand = torch.rand(B, seq_len, device=device)
        mask = rand < target_mask_ratio.unsqueeze(1)  # [B, seq_len]

        # Do not mask padding positions
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        valid = pos_idx < lengths.unsqueeze(1)
        mask = mask & valid

        # Ensure at least one masked position per sequence (if any valid positions)
        for b in range(B):
            if not mask[b].any() and lengths[b] > 0:
                valid_positions = valid[b].nonzero(as_tuple=True)[0]
                if valid_positions.numel() > 0:
                    j = torch.randint(valid_positions.numel(), (1,), device=device)
                    mask[b, valid_positions[j]] = True

        masked_actions = actions.clone()
        masked_actions[mask] = self.model.mask_token_id

        actual_mask_ratio = mask.float().sum(dim=1) / lengths.float().clamp(min=1)

        return masked_actions, mask, actual_mask_ratio
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Dictionary with 'states' and 'actions'
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Unpack batch
        states = {k: v.to(self.device) for k, v in batch['states'].items()}
        actions = batch['actions'].to(self.device)  # [B, seq_len]
        lengths = batch.get('length', torch.full_like(actions[:, 0], actions.shape[1])).to(self.device)
        
        B, seq_len = actions.shape
        max_seq_len = self.model.max_seq_len

        # Pad / truncate discrete actions to max_seq_len
        if seq_len < max_seq_len:
            pad_value = 0  # treat as some valid (but rarely used) action
            padding = torch.full(
                (B, max_seq_len - seq_len),
                pad_value,
                dtype=torch.long,
                device=self.device,
            )
            actions_padded = torch.cat([actions, padding], dim=1)
        elif seq_len > max_seq_len:
            actions_padded = actions[:, :max_seq_len]
            lengths = lengths.clamp(max=max_seq_len)
        else:
            actions_padded = actions

        # Create masked input using MASK token, not Gaussian noise
        masked_actions, mask, mask_ratio = self.create_masked_input(
            actions_padded, lengths
        )

        # Forward pass: model embeds masked_actions internally
        logits = self.model(masked_actions, states, mask_ratio)  # [B, max_seq_len, num_actions]
        
        # Compute loss only on masked positions
        loss_per_token = self.criterion(
            logits.reshape(-1, self.model.num_actions),
            actions_padded.reshape(-1),
        ).reshape(B, max_seq_len)

        loss = (loss_per_token * mask.float()).sum() / mask.sum().clamp(min=1)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Metrics
        with torch.no_grad():
            pred_actions = logits.argmax(dim=-1)
            correct = ((pred_actions == actions_padded) & mask).sum().float()
            accuracy = correct / mask.sum().clamp(min=1)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'mask_ratio': mask_ratio.mean().item(),
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        metrics_sum = {'loss': 0.0, 'accuracy': 0.0, 'mask_ratio': 0.0}
        
        for batch in pbar:
            metrics = self.train_step(batch)
            
            for k, v in metrics.items():
                metrics_sum[k] += v
            
            # Update progress bar
            avg_metrics = {k: v / (pbar.n + 1) for k, v in metrics_sum.items()}
            pbar.set_postfix(avg_metrics)
        
        return {k: v / len(self.train_loader) for k, v in metrics_sum.items()}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        metrics_sum = {'loss': 0.0, 'accuracy': 0.0}
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            states = {k: v.to(self.device) for k, v in batch['states'].items()}
            actions = batch['actions'].to(self.device)
            lengths = batch.get('length', torch.full_like(actions[:, 0], actions.shape[1])).to(self.device)
            
            B, seq_len = actions.shape
            max_seq_len = self.model.max_seq_len  # Use max_seq_len, not num_tokens!
            
            # Pad/truncate actions to max_seq_len
            if seq_len < max_seq_len:
                pad_value = 0
                padding = torch.full(
                    (B, max_seq_len - seq_len),
                    pad_value,
                    dtype=torch.long,
                    device=self.device,
                )
                actions_padded = torch.cat([actions, padding], dim=1)
            elif seq_len > max_seq_len:
                actions_padded = actions[:, :max_seq_len]
                lengths = lengths.clamp(max=max_seq_len)
            else:
                actions_padded = actions

            # Use fixed 50% mask ratio for validation
            rand = torch.rand(B, max_seq_len, device=self.device)
            mask = rand < 0.5

            pos_idx = torch.arange(max_seq_len, device=self.device).unsqueeze(0)
            valid = pos_idx < lengths.unsqueeze(1)
            mask = mask & valid

            masked_actions = actions_padded.clone()
            masked_actions[mask] = self.model.mask_token_id

            mask_ratio = (mask.float().sum(dim=1) / lengths.float().clamp(min=1))

            # Forward
            logits = self.model(masked_actions, states, mask_ratio)
            
            # Loss
            loss_per_token = self.criterion(
                logits.reshape(-1, self.model.num_actions),
                actions_padded.reshape(-1),
            ).reshape(B, max_seq_len)

            loss = (loss_per_token * mask.float()).sum() / mask.sum().clamp(min=1)
            
            # Accuracy
            pred_actions = logits.argmax(dim=-1)
            correct = ((pred_actions == actions_padded) & mask).sum().float()
            accuracy = correct / mask.sum().clamp(min=1)
            
            metrics_sum['loss'] += loss.item()
            metrics_sum['accuracy'] += accuracy.item()
        
        return {k: v / len(self.val_loader) for k, v in metrics_sum.items()}
    
    def train(self, num_epochs: int, logger=None):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            logger: Optional logger (wandb, tensorboard, etc.)
        """
        best_val_loss = float('inf')
        
        # Store num_epochs for curriculum calculation
        if self.curriculum_warmup_epochs is None:
            self.curriculum_warmup_epochs = num_epochs
        
        for epoch in range(num_epochs):
            # Update current epoch for curriculum learning
            self.current_epoch = epoch
            
            # Get current curriculum mask ratio for logging
            if self.use_curriculum:
                current_target_ratio = self.get_curriculum_mask_ratio()
            else:
                current_target_ratio = None
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.evaluate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            metrics = {
                'epoch': epoch,
                'lr': current_lr,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
            }
            
            if current_target_ratio is not None:
                metrics['train/target_mask_ratio'] = current_target_ratio
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}")
            if current_target_ratio is not None:
                print(f"  Target mask ratio: {current_target_ratio:.3f} (actual: {train_metrics['mask_ratio']:.3f})")
            print(f"  Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")
            print(f"  LR:    {current_lr:.6f}")
            
            if logger:
                logger.log(metrics)
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                if hasattr(self, 'checkpoint_dir'):
                    # Save checkpoint with config for reproducibility
                    checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_metrics['loss'],
                        'config': self.config,  # Include config for easy loading
                    }
                    # If experiment_config is stored, use it for more detailed config
                    if hasattr(self, 'experiment_config'):
                        checkpoint['experiment_config'] = self.experiment_config.to_dict()
                    torch.save(
                        checkpoint,
                        f"{self.checkpoint_dir}/best_model_epoch{epoch}.pt"
                    )
