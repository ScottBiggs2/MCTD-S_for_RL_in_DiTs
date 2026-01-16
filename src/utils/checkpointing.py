"""
Model checkpointing utilities.
"""
import torch
from pathlib import Path
from typing import Dict, Optional, Any
import json


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
    filename: str = "checkpoint.pt",
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        **kwargs: Additional metadata to save
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / filename
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        **kwargs
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    # Also save metadata as JSON
    metadata = {
        'epoch': epoch,
        'metrics': metrics,
        **{k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
    }
    
    metadata_path = checkpoint_dir / filename.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_checkpoint(
    checkpoint_path: Path,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into (optional)
        optimizer: Optimizer to load state into (optional)
        device: Device to load checkpoint on
    
    Returns:
        Dictionary with checkpoint contents
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model state from {checkpoint_path}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded optimizer state from {checkpoint_path}")
    
    return checkpoint
