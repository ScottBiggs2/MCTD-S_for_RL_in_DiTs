"""
Logging utilities for training (wandb, tensorboard).
"""
from typing import Dict, Optional, Any
import os


class Logger:
    """
    Simple logger interface that can use wandb, tensorboard, or print.
    """
    def __init__(self, use_wandb: bool = False, project_name: str = "maze-mctd"):
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.wandb_run = None
        
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=project_name,
                    name=f"baseline-{os.environ.get('USER', 'user')}",
                )
            except ImportError:
                print("Warning: wandb not available, using print logging")
                self.use_wandb = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.log(metrics, step=step)
        else:
            # Print logging
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
    
    def finish(self):
        """Finish logging."""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()
