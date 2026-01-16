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
    
    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        """
        Cosine masking schedule: mask_ratio = cos(π/2 * (1-t))
        
        Args:
            t: [batch] timestep values in [0, 1]
        
        Returns:
            [batch] mask ratios in [0, 1]
        """
        schedule = self.config.get('mask_schedule', 'cosine')
        
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
    
    def create_mask(
        self,
        batch_size: int,
        seq_len: int,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Create random mask with ratio determined by timestep.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            t: [batch] timestep values
        
        Returns:
            [batch, seq_len] binary mask (1 = masked position)
        """
        mask_ratio = self.get_mask_ratio(t)  # [batch]
        
        # Random masks for each sequence
        masks = []
        for i in range(batch_size):
            num_masked = int(seq_len * mask_ratio[i].item())
            num_masked = max(1, min(num_masked, seq_len - 1))  # At least 1, at most seq_len-1
            
            # Random positions to mask
            mask_indices = torch.randperm(seq_len, device=self.device)[:num_masked]
            mask = torch.zeros(seq_len, device=self.device, dtype=torch.bool)
            mask[mask_indices] = True
            masks.append(mask)
        
        return torch.stack(masks)  # [batch, seq_len]
    
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
        
        B, seq_len = actions.shape
        num_tokens = self.model.num_tokens  # 49
        
        # Sample random timestep
        t = torch.rand(B, device=self.device)  # [B] in [0, 1]
        
        # Encode actions to hidden space
        clean_hidden = self.model.action_encoder(actions)  # [B, seq_len, hidden_dim]
        
        # Pad/truncate to num_tokens if needed
        if seq_len != num_tokens:
            if seq_len < num_tokens:
                # Pad with last action
                padding = clean_hidden[:, -1:, :].expand(B, num_tokens - seq_len, -1)
                clean_hidden = torch.cat([clean_hidden, padding], dim=1)
                actions_padded = torch.cat([
                    actions,
                    actions[:, -1:].expand(B, num_tokens - seq_len)
                ], dim=1)
            else:
                # Truncate to num_tokens
                clean_hidden = clean_hidden[:, :num_tokens, :]
                actions_padded = actions[:, :num_tokens]
        
        # Create mask
        mask = self.create_mask(B, num_tokens, t)  # [B, num_tokens]
        
        # Add noise to masked positions
        noise = torch.randn_like(clean_hidden)
        noisy_hidden = torch.where(
            mask[..., None],
            noise,
            clean_hidden
        )
        
        # Forward pass
        logits = self.model(noisy_hidden, states, t, mask)  # [B, num_tokens, num_actions]
        
        # Compute loss only on masked positions
        loss_per_token = self.criterion(
            logits.reshape(-1, self.model.num_actions),
            actions_padded.reshape(-1)
        ).reshape(B, num_tokens)
        
        loss = (loss_per_token * mask.float()).sum() / mask.sum().clamp(min=1)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Metrics
        with torch.no_grad():
            pred_actions = logits.argmax(dim=-1)
            accuracy = ((pred_actions == actions_padded) * mask).sum() / mask.sum()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'mask_ratio': mask.float().mean().item(),
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
            
            B, seq_len = actions.shape
            num_tokens = self.model.num_tokens
            
            # Fixed timestep for evaluation
            t = torch.ones(B, device=self.device) * 0.5
            
            # Encode actions
            clean_hidden = self.model.action_encoder(actions)
            
            # Pad/truncate
            if seq_len != num_tokens:
                if seq_len < num_tokens:
                    padding = clean_hidden[:, -1:, :].expand(B, num_tokens - seq_len, -1)
                    clean_hidden = torch.cat([clean_hidden, padding], dim=1)
                    actions_padded = torch.cat([
                        actions,
                        actions[:, -1:].expand(B, num_tokens - seq_len)
                    ], dim=1)
                else:
                    clean_hidden = clean_hidden[:, :num_tokens, :]
                    actions_padded = actions[:, :num_tokens]
            
            # Create mask
            mask = self.create_mask(B, num_tokens, t)
            
            # Add noise
            noise = torch.randn_like(clean_hidden)
            noisy_hidden = torch.where(mask[..., None], noise, clean_hidden)
            
            # Forward
            logits = self.model(noisy_hidden, states, t, mask)
            
            # Loss
            loss_per_token = self.criterion(
                logits.reshape(-1, self.model.num_actions),
                actions_padded.reshape(-1)
            ).reshape(B, num_tokens)
            
            loss = (loss_per_token * mask.float()).sum() / mask.sum().clamp(min=1)
            
            # Accuracy
            pred_actions = logits.argmax(dim=-1)
            accuracy = ((pred_actions == actions_padded) * mask).sum() / mask.sum()
            
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
        
        for epoch in range(num_epochs):
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
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}")
            print(f"  Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}")
            print(f"  LR:    {current_lr:.6f}")
            
            if logger:
                logger.log(metrics)
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                if hasattr(self, 'checkpoint_dir'):
                    torch.save(
                        self.model.state_dict(),
                        f"{self.checkpoint_dir}/best_model_epoch{epoch}.pt"
                    )
