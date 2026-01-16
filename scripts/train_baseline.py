"""
Training script for baseline masked diffusion policy.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path
import pickle
from torch.utils.data import DataLoader

from src.models.diffusion_policy import DiffusionPolicy
from src.environments.trajectory_dataset import TrajectoryDataset
from src.training.mdlm_trainer import MaskedDiffusionTrainer
from src.utils.logging import Logger
from src.utils.checkpointing import save_checkpoint


def load_dataset(data_path, max_seq_len=64):
    """Load trajectory dataset from pickle file."""
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    return TrajectoryDataset(trajectories, max_seq_len=max_seq_len)


def main():
    """Main training function."""
    # Configuration
    config = {
        # Model
        'num_actions': 7,
        'hidden_dim': 64,
        'num_layers': 4,
        'num_heads': 4,
        'num_tokens': 49,  # 7x7 grid
        'max_seq_len': 64,
        'dropout': 0.1,
        
        # Training
        'batch_size': 16,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,  # Can use 1e-5 if needed
        'num_epochs': 3,
        'num_diffusion_steps': 100,
        'mask_schedule': 'cosine',
        'lr_patience': 5,
        
        # Data
        'data_dir': 'data',
        'env_name': 'Empty-8x8',  # or 'FourRooms'
        
        # Checkpoints
        'checkpoint_dir': 'checkpoints',
        
        # Logging
        'use_wandb': False,  # Set to True if wandb installed
    }
    
    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load datasets
    data_dir = Path(config['data_dir'])
    env_name = config['env_name']
    
    train_path = data_dir / f"{env_name}_train.pkl"
    test_path = data_dir / f"{env_name}_test.pkl"
    
    print(f"Loading train data from {train_path}")
    train_dataset = load_dataset(train_path, max_seq_len=config['max_seq_len'])
    
    print(f"Loading test data from {test_path}")
    val_dataset = load_dataset(test_path, max_seq_len=config['max_seq_len'])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # M1 MacBook
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    
    # Model
    print("\nInitializing model...")
    model = DiffusionPolicy(
        num_actions=config['num_actions'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        num_tokens=config['num_tokens'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Trainer
    print("\nInitializing trainer...")
    trainer = MaskedDiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    
    # Set checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.checkpoint_dir = checkpoint_dir
    
    # Logger
    logger = None
    if config.get('use_wandb', False):
        logger = Logger(use_wandb=True, project_name="maze-mctd")
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training for {config['num_epochs']} epochs")
    print(f"{'='*60}")
    
    trainer.train(
        num_epochs=config['num_epochs'],
        logger=logger
    )
    
    # Finish logging
    if logger:
        logger.finish()
    
    print("\nTraining complete!")
    print(f"Best model saved to: {checkpoint_dir}/")


if __name__ == "__main__":
    main()
