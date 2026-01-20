"""
Pretraining trainer for StateCNNEncoder with reconstruction and auxiliary tasks.

Implements weighted reconstruction loss:
- Initial/goal positions: highest weight
- Walls: medium weight
- Background: lowest weight
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from ..models.state_cnn_encoder import StateCNNEncoder


def create_weight_mask(grid: torch.Tensor, agent_pos: Optional[torch.Tensor] = None, 
                      goal_pos: Optional[torch.Tensor] = None, grid_size: int = 19) -> torch.Tensor:
    """
    Create weight mask for reconstruction loss.
    
    Weights:
    - Initial/goal positions: 10.0
    - Walls: 3.0
    - Background: 1.0
    
    Args:
        grid: [B, H, W, 3] or [B, 3, H, W] grid observation
        agent_pos: [B, 2] agent positions (x, y) or None
        goal_pos: [B, 2] goal positions (x, y) or None
        grid_size: Grid size (H = W = grid_size)
    
    Returns:
        weight_mask: [B, H, W] weight mask
    """
    B = grid.shape[0]
    
    # Handle different input formats
    if grid.dim() == 4:
        if grid.shape[1] == 3:
            # [B, 3, H, W] - channels first
            grid_spatial = grid.permute(0, 2, 3, 1)  # [B, H, W, 3]
        else:
            # [B, H, W, 3] - channels last
            grid_spatial = grid
    else:
        raise ValueError(f"Unexpected grid shape: {grid.shape}")
    
    H, W = grid_spatial.shape[1], grid_spatial.shape[2]
    
    # Initialize with background weight (1.0)
    weight_mask = torch.ones(B, H, W, device=grid.device, dtype=grid.dtype)
    
    # Mark walls (object type == 2) with weight 3.0
    # Channel 0 contains object type
    obj_type = grid_spatial[:, :, :, 0]  # [B, H, W]
    wall_mask = (obj_type == 2).float()  # Walls have object type 2
    weight_mask = weight_mask + wall_mask * 2.0  # Add 2.0 to get 3.0 total
    
    # Mark agent and goal positions with weight 10.0
    if agent_pos is not None:
        agent_pos = agent_pos.long()  # Ensure integer indices
        for b in range(B):
            x, y = agent_pos[b, 0].item(), agent_pos[b, 1].item()
            if 0 <= x < W and 0 <= y < H:
                weight_mask[b, y, x] = 10.0  # Note: y is first dimension in image
    
    if goal_pos is not None:
        goal_pos = goal_pos.long()
        for b in range(B):
            x, y = goal_pos[b, 0].item(), goal_pos[b, 1].item()
            if 0 <= x < W and 0 <= y < H:
                weight_mask[b, y, x] = 10.0
    
    return weight_mask


class StateEncoderPretrainer:
    """
    Pretraining trainer for StateCNNEncoder.
    
    Trains with:
    1. Weighted reconstruction loss (grid -> grid)
    2. Agent position prediction
    3. Goal position prediction
    4. Direction prediction
    """
    def __init__(
        self,
        model: StateCNNEncoder,
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
        
        # Loss functions
        self.reconstruction_criterion = nn.MSELoss(reduction='none')
        self.position_criterion = nn.MSELoss()
        self.direction_criterion = nn.CrossEntropyLoss()
        
        # Loss weights
        self.reconstruction_weight = config.get('reconstruction_weight', 1.0)
        self.agent_pos_weight = config.get('agent_pos_weight', 1.0)
        self.goal_pos_weight = config.get('goal_pos_weight', 1.0)
        self.direction_weight = config.get('direction_weight', 1.0)
        
        self.grid_size = config.get('grid_size', 19)
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Unpack batch
        grid = batch['grid'].to(self.device)  # [B, H, W, 3] or [B, H*W*3]
        direction = batch['direction'].to(self.device)  # [B]
        
        # Get agent and goal positions if available
        agent_pos = batch.get('agent_pos')
        goal_pos = batch.get('goal_pos')
        if agent_pos is not None:
            agent_pos = agent_pos.to(self.device)
        if goal_pos is not None:
            goal_pos = goal_pos.to(self.device)
        
        # Prepare state dict
        state_dict = {'grid': grid, 'direction': direction}
        
        # Forward pass with latent (returns embedding, latent)
        embedding, latent = self.model.forward_with_features(state_dict)
        
        # Reconstruction loss
        reconstructed_grid = self.model.decode(latent)  # [B, 3, H, W]
        
        # Prepare target grid for comparison
        if grid.dim() == 2:
            # Flattened: [B, H*W*3] -> [B, H, W, 3]
            B = grid.shape[0]
            grid_reshaped = grid.reshape(B, self.grid_size, self.grid_size, 3)
            grid_reshaped = grid_reshaped.permute(0, 3, 1, 2)  # [B, 3, H, W]
        elif grid.dim() == 4:
            if grid.shape[1] == 3:
                grid_reshaped = grid  # Already [B, 3, H, W]
            else:
                grid_reshaped = grid.permute(0, 3, 1, 2)  # [B, H, W, 3] -> [B, 3, H, W]
        else:
            raise ValueError(f"Unexpected grid shape: {grid.shape}")
        
        # Normalize target to [0, 1] for comparison with sigmoid output
        # Grid values are indices (0-10 range typically, max ~11 for MiniGrid)
        # Normalize by max possible value (11) to get [0, 1] range
        target_grid = grid_reshaped.float() / 11.0  # Normalize indices [0-11] to [0-1]
        
        # Weighted reconstruction loss
        recon_loss_per_pixel = self.reconstruction_criterion(
            reconstructed_grid, target_grid
        )  # [B, 3, H, W]
        
        # Create weight mask (handle None positions)
        # Need to pass channels-last format for weight mask function
        # grid_reshaped is [B, 3, H, W], but create_weight_mask expects [B, H, W, 3]
        grid_for_mask = grid_reshaped.permute(0, 2, 3, 1)  # [B, 3, H, W] -> [B, H, W, 3]
        agent_pos_for_mask = agent_pos if agent_pos is not None else None
        goal_pos_for_mask = goal_pos if goal_pos is not None else None
        weight_mask = create_weight_mask(
            grid_for_mask, agent_pos_for_mask, goal_pos_for_mask, self.grid_size
        )  # [B, H, W]
        weight_mask = weight_mask.unsqueeze(1).expand(-1, 3, -1, -1)  # [B, 3, H, W]
        
        # Apply weights and average
        weighted_recon_loss = (recon_loss_per_pixel * weight_mask).mean()
        
        # Auxiliary prediction losses
        aux_predictions = self.model.predict_auxiliary(embedding)
        
        total_loss = self.reconstruction_weight * weighted_recon_loss
        metrics = {
            'reconstruction_loss': weighted_recon_loss.item(),
        }
        
        # Agent position loss
        if agent_pos is not None:
            agent_pos_loss = self.position_criterion(
                aux_predictions['agent_pos'], agent_pos.float()
            )
            total_loss += self.agent_pos_weight * agent_pos_loss
            metrics['agent_pos_loss'] = agent_pos_loss.item()
        
        # Goal position loss
        if goal_pos is not None:
            goal_pos_loss = self.position_criterion(
                aux_predictions['goal_pos'], goal_pos.float()
            )
            total_loss += self.goal_pos_weight * goal_pos_loss
            metrics['goal_pos_loss'] = goal_pos_loss.item()
        
        # Direction loss
        direction_loss = self.direction_criterion(
            aux_predictions['direction'], direction
        )
        total_loss += self.direction_weight * direction_loss
        metrics['direction_loss'] = direction_loss.item()
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        metrics['total_loss'] = total_loss.item()
        
        return metrics
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        metrics_sum = {}
        
        for batch in pbar:
            metrics = self.train_step(batch)
            
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            
            # Update progress bar
            avg_metrics = {k: v / (pbar.n + 1) for k, v in metrics_sum.items()}
            pbar.set_postfix(avg_metrics)
        
        return {k: v / len(self.train_loader) for k, v in metrics_sum.items()}
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        metrics_sum = {}
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            # Similar to train_step but no backward
            grid = batch['grid'].to(self.device)
            direction = batch['direction'].to(self.device)
            agent_pos = batch.get('agent_pos')
            goal_pos = batch.get('goal_pos')
            if agent_pos is not None:
                agent_pos = agent_pos.to(self.device)
            if goal_pos is not None:
                goal_pos = goal_pos.to(self.device)
            
            # Prepare state dict
            state_dict = {'grid': grid, 'direction': direction}
            
            # Forward pass
            embedding, latent = self.model.forward_with_features(state_dict)
            
            # Reconstruction
            reconstructed_grid = self.model.decode(latent)
            
            # Prepare target
            if grid.dim() == 2:
                B = grid.shape[0]
                grid_reshaped = grid.reshape(B, self.grid_size, self.grid_size, 3)
                grid_reshaped = grid_reshaped.permute(0, 3, 1, 2)
            elif grid.dim() == 4:
                if grid.shape[1] == 3:
                    grid_reshaped = grid
                else:
                    grid_reshaped = grid.permute(0, 3, 1, 2)
            
            target_grid = grid_reshaped.float() / 11.0  # Normalize indices [0-11] to [0-1]
            
            # Losses
            recon_loss_per_pixel = self.reconstruction_criterion(
                reconstructed_grid, target_grid
            )
            # Need to pass channels-last format for weight mask function
            grid_for_mask = grid_reshaped.permute(0, 2, 3, 1)  # [B, 3, H, W] -> [B, H, W, 3]
            agent_pos_for_mask = agent_pos if agent_pos is not None else None
            goal_pos_for_mask = goal_pos if goal_pos is not None else None
            weight_mask = create_weight_mask(
                grid_for_mask, agent_pos_for_mask, goal_pos_for_mask, self.grid_size
            ).unsqueeze(1).expand(-1, 3, -1, -1)
            weighted_recon_loss = (recon_loss_per_pixel * weight_mask).mean()
            
            aux_predictions = self.model.predict_auxiliary(embedding)
            
            total_loss = self.reconstruction_weight * weighted_recon_loss
            batch_metrics = {
                'reconstruction_loss': weighted_recon_loss.item(),
            }
            
            if agent_pos is not None:
                agent_pos_loss = self.position_criterion(
                    aux_predictions['agent_pos'], agent_pos.float()
                )
                total_loss += self.agent_pos_weight * agent_pos_loss
                batch_metrics['agent_pos_loss'] = agent_pos_loss.item()
            
            if goal_pos is not None:
                goal_pos_loss = self.position_criterion(
                    aux_predictions['goal_pos'], goal_pos.float()
                )
                total_loss += self.agent_pos_weight * goal_pos_loss
                batch_metrics['goal_pos_loss'] = goal_pos_loss.item()
            
            direction_loss = self.direction_criterion(
                aux_predictions['direction'], direction
            )
            total_loss += self.direction_weight * direction_loss
            batch_metrics['direction_loss'] = direction_loss.item()
            batch_metrics['total_loss'] = total_loss.item()
            
            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
        
        return {k: v / len(self.val_loader) for k, v in metrics_sum.items()}
    
    @torch.no_grad()
    def visualize_reconstruction(
        self, 
        batch: Dict, 
        save_path: Optional[Path] = None,
        num_samples: int = 5
    ) -> plt.Figure:
        """
        Visualize true vs reconstructed grids.
        
        Grid values are indices (object_type, color, state), not RGB.
        We'll visualize each channel separately and also create a decoded RGB visualization.
        
        Args:
            batch: Batch of data
            save_path: Optional path to save figure
            num_samples: Number of samples to visualize
        
        Returns:
            matplotlib figure
        """
        import matplotlib.colors as mcolors
        import numpy as np
        
        self.model.eval()
        
        grid = batch['grid'].to(self.device)
        direction = batch['direction'].to(self.device)
        state_dict = {'grid': grid, 'direction': direction}
        
        embedding, latent = self.model.forward_with_features(state_dict)
        reconstructed_grid = self.model.decode(latent)  # [B, 3, H, W]
        
        # Prepare true grid
        if grid.dim() == 2:
            B = grid.shape[0]
            grid_reshaped = grid.reshape(B, self.grid_size, self.grid_size, 3)
            grid_reshaped = grid_reshaped.permute(0, 3, 1, 2)  # [B, 3, H, W]
        elif grid.dim() == 4:
            if grid.shape[1] == 3:
                grid_reshaped = grid
            else:
                grid_reshaped = grid.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected grid dim: {grid.dim()}")
        
        # Grid values are uint8 indices (0-255 range), but actual values are 0-10 range
        # Convert to numpy and ensure proper type
        true_grid_np = grid_reshaped.cpu().numpy().astype(np.uint8)  # [B, 3, H, W]
        
        # Denormalize reconstructed grid (it's in [0, 1] from sigmoid)
        # Multiply by 11 to get back to index range [0-11], then clip and convert to uint8
        recon_grid_np = (reconstructed_grid.cpu().numpy() * 11.0).clip(0, 11).astype(np.uint8)  # [B, 3, H, W]
        
        # Convert to visual representation
        # Channel 0: object type (0=unseen, 1=empty, 2=wall, 8=goal, 10=agent)
        # Channel 1: color (0-5)
        # Channel 2: state (0-1, or direction for agent)
        
        def decode_grid_to_rgb(grid_chw: np.ndarray) -> np.ndarray:
            """Convert MiniGrid encoding to RGB visualization."""
            B, C, H, W = grid_chw.shape
            rgb = np.zeros((B, H, W, 3), dtype=np.uint8)
            
            obj_type = grid_chw[:, 0, :, :]  # [B, H, W]
            color_idx = grid_chw[:, 1, :, :]  # [B, H, W]
            state_idx = grid_chw[:, 2, :, :]  # [B, H, W]
            
            # Color map for object types
            obj_colors = {
                0: (0, 0, 0),      # unseen - black
                1: (255, 255, 255), # empty - white
                2: (128, 128, 128), # wall - gray
                4: (255, 255, 0),   # door - yellow
                5: (255, 165, 0),   # key - orange
                8: (0, 255, 0),     # goal - green
                10: (255, 0, 0),    # agent - red
            }
            
            # Color map for colors (if object has color)
            color_map = {
                0: (255, 0, 0),    # red
                1: (0, 255, 0),    # green
                2: (0, 0, 255),    # blue
                3: (128, 0, 128),  # purple
                4: (255, 255, 0),  # yellow
                5: (128, 128, 128), # grey
            }
            
            for b in range(B):
                for h in range(H):
                    for w in range(W):
                        obj = obj_type[b, h, w]
                        col = color_idx[b, h, w]
                        
                        if obj == 1:  # empty
                            rgb[b, h, w] = (255, 255, 255)  # white
                        elif obj == 2:  # wall
                            rgb[b, h, w] = (128, 128, 128)  # gray
                        elif obj == 8:  # goal
                            rgb[b, h, w] = (0, 255, 0)  # green
                        elif obj == 10:  # agent
                            rgb[b, h, w] = (255, 0, 0)  # red
                        elif obj == 4:  # door
                            if col in color_map:
                                rgb[b, h, w] = tuple(int(c * 0.7) for c in color_map[col])  # darker
                            else:
                                rgb[b, h, w] = (255, 255, 0)  # yellow
                        elif obj in obj_colors:
                            rgb[b, h, w] = obj_colors[obj]
                        else:
                            # Use color index if available
                            if col in color_map:
                                rgb[b, h, w] = color_map[col]
                            else:
                                rgb[b, h, w] = (200, 200, 200)  # light gray
            
            return rgb
        
        # Convert to RGB
        true_rgb = decode_grid_to_rgb(true_grid_np)  # [B, H, W, 3]
        recon_rgb = decode_grid_to_rgb(recon_grid_np)  # [B, H, W, 3]
        
        # Visualize
        num_samples = min(num_samples, grid.shape[0])
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # True grid (RGB)
            axes[i, 0].imshow(true_rgb[i], interpolation='nearest', vmin=0, vmax=255)
            axes[i, 0].set_title(f'Sample {i+1}: True Grid')
            axes[i, 0].axis('off')
            
            # Reconstructed grid (RGB)
            axes[i, 1].imshow(recon_rgb[i], interpolation='nearest', vmin=0, vmax=255)
            axes[i, 1].set_title(f'Sample {i+1}: Reconstructed Grid')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def train(self, num_epochs: int, logger=None, viz_dir: Optional[Path] = None):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            logger: Optional logger (wandb, tensorboard, etc.)
            viz_dir: Optional directory to save visualizations
        """
        best_val_loss = float('inf')
        
        if viz_dir:
            viz_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.evaluate()
            
            # Update learning rate
            self.scheduler.step(val_metrics['total_loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            metrics = {
                'epoch': epoch,
                'lr': current_lr,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
            }
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train: loss={train_metrics['total_loss']:.4f}")
            print(f"  Val:   loss={val_metrics['total_loss']:.4f}")
            print(f"  LR:    {current_lr:.6f}")
            
            if logger:
                logger.log(metrics)
            
            # Visualize on validation set (every epoch)
            if viz_dir:
                # Get a batch from validation set
                val_batch = next(iter(self.val_loader))
                viz_path = viz_dir / f'reconstruction_epoch_{epoch}.png'
                self.visualize_reconstruction(val_batch, save_path=viz_path)
                print(f"  Saved visualization to {viz_path}")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                if hasattr(self, 'checkpoint_dir'):
                    checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_metrics['total_loss'],
                        'config': self.config,
                    }
                    torch.save(
                        checkpoint,
                        f"{self.checkpoint_dir}/best_state_encoder_epoch{epoch}.pt"
                    )
