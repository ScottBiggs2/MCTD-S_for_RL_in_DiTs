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
from src.config import get_experiment_config


def load_dataset(data_path, max_seq_len=64, use_augmentation=True, grid_size=19):
    """Load trajectory dataset from pickle file."""
    with open(data_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    return TrajectoryDataset(
        trajectories, 
        max_seq_len=max_seq_len,
        use_augmentation=use_augmentation,
        grid_size=grid_size,
    )


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline masked diffusion policy')
    parser.add_argument('--pretrained_state_encoder', type=str, default=None,
                       help='Path to pretrained state encoder checkpoint (from Stage 1 pretraining)')
    parser.add_argument('--freeze_state_encoder', action='store_true',
                       help='Freeze state encoder during training (Stage 2: train DiT only)')
    parser.add_argument('--env', type=str, default=None,
                       help='Environment name (overrides config default)')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Hidden dimension (must match pretrained encoder if loading one)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu, cuda, mps, or auto)')
    
    args = parser.parse_args()
    
    # Load configuration from central config module
    # This ensures consistency across all scripts
    experiment_config = get_experiment_config()
    
    # Convert to dict format for compatibility with existing code
    config = experiment_config.to_dict()
    
    # Override with command-line arguments
    if args.pretrained_state_encoder:
        config['pretrained_state_encoder_path'] = args.pretrained_state_encoder
    if args.freeze_state_encoder:
        config['freeze_state_encoder'] = True
    if args.env:
        config['env_name'] = args.env
    if args.hidden_dim is not None:
        config['hidden_dim'] = args.hidden_dim
        # Also update the model config object
        experiment_config.model.hidden_dim = args.hidden_dim
    
    # Override environment name if needed (can be set via config or kept as default)
    # config['env_name'] = 'FourRooms'  # Uncomment to change environment
    
    # Device
    if args.device:
        if args.device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = args.device
    else:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load datasets
    data_dir = Path(config['data_dir'])
    env_name = config['env_name']
    
    train_path = data_dir / f"{env_name}_train.pkl"
    test_path = data_dir / f"{env_name}_test.pkl"
    
    # Determine grid size from environment name
    # FourRooms is 19x19, Empty-8x8 is 8x8, etc.
    grid_size_map = {
        'MiniGrid-FourRooms-v0': 19,
        'MiniGrid-Empty-8x8-v0': 8,
        'FourRooms': 19,
        'Empty-8x8': 8,
    }
    grid_size = grid_size_map.get(env_name, 19)  # Default to 19 for FourRooms
    
    print(f"Loading train data from {train_path}")
    train_dataset = load_dataset(
        train_path, 
        max_seq_len=config['max_seq_len'],
        use_augmentation=True,  # Enable augmentation for 4x effective data
        grid_size=grid_size,
    )
    
    print(f"Loading test data from {test_path}")
    val_dataset = load_dataset(
        test_path, 
        max_seq_len=config['max_seq_len'],
        use_augmentation=False,  # No augmentation on validation set
        grid_size=grid_size,
    )
    
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
    
    # Load pretrained state encoder if specified
    pretrained_state_encoder_path = config.get('pretrained_state_encoder_path')
    if pretrained_state_encoder_path:
        pretrained_path = Path(pretrained_state_encoder_path)
        if not pretrained_path.exists():
            print(f"\n⚠️  Warning: Pretrained state encoder path '{pretrained_state_encoder_path}' not found!")
            print("  Continuing without pretrained encoder (training from scratch)...")
        else:
            print(f"\n{'='*60}")
            print(f"Loading pretrained state encoder from {pretrained_state_encoder_path}")
            print(f"{'='*60}")
            checkpoint = torch.load(pretrained_state_encoder_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                checkpoint_config = checkpoint.get('config', {})
            else:
                state_dict = checkpoint
                checkpoint_config = {}
            
            # Check for hidden_dim mismatch
            if checkpoint_config:
                pretrained_hidden_dim = checkpoint_config.get('hidden_dim')
                current_hidden_dim = config.get('hidden_dim')
                if pretrained_hidden_dim and pretrained_hidden_dim != current_hidden_dim:
                    print(f"\n⚠️  WARNING: Hidden dimension mismatch!")
                    print(f"  Pretrained encoder: hidden_dim={pretrained_hidden_dim}")
                    print(f"  Current model: hidden_dim={current_hidden_dim}")
                    print(f"  This will cause loading errors!")
                    print(f"\n  Solution: Set --hidden_dim {pretrained_hidden_dim} or update config")
                    raise ValueError(
                        f"Hidden dimension mismatch: pretrained={pretrained_hidden_dim}, "
                        f"current={current_hidden_dim}. Update config or use matching hidden_dim."
                    )
            
            # Load state encoder weights (may have decoder/auxiliary heads we don't need)
            # Filter to only load encoder weights
            encoder_state_dict = {}
            for k, v in state_dict.items():
                # Skip decoder and auxiliary heads
                if 'decoder' not in k and 'agent_pos_head' not in k and 'goal_pos_head' not in k and 'direction_head' not in k:
                    encoder_state_dict[k] = v
            
            # Load into model's state encoder
            model.state_encoder.load_state_dict(encoder_state_dict, strict=False)
            print("✓ Pretrained state encoder loaded (decoder/auxiliary heads ignored)")
            
            # Optionally freeze state encoder for Stage 2 training
            if config.get('freeze_state_encoder', False):
                print("\nFreezing state encoder for Stage 2 training (DiT + action encoder only)...")
                for param in model.state_encoder.parameters():
                    param.requires_grad = False
                print("✓ State encoder frozen")
                print("  Note: Only DiT blocks and action encoder will be trained")
            else:
                print("\nState encoder is trainable (will fine-tune end-to-end)")
    else:
        print("\n" + "="*60)
        print("Training from scratch (no pretrained state encoder)")
        print("="*60)
        print("  To use pretrained state encoder:")
        print("    python scripts/train_baseline.py --pretrained_state_encoder <path>")
        print("  To freeze state encoder (Stage 2):")
        print("    python scripts/train_baseline.py --pretrained_state_encoder <path> --freeze_state_encoder")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    if frozen_params > 0:
        print(f"  Frozen: {frozen_params:,} (state encoder)")
    
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
    
    # Store experiment config in trainer for checkpoint saving
    trainer.experiment_config = experiment_config
    
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
