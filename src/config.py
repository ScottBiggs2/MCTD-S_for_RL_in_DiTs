"""
Centralized configuration for Maze MCTD experiments.

This module provides default configurations for model architecture, training,
and evaluation to ensure consistency across scripts.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Action space
    num_actions: int = 7  # MiniGrid standard action space
    
    # Transformer architecture
    hidden_dim: int = 256  # Hidden dimension for embeddings and transformer layers
    num_layers: int = 4  # Number of DiT transformer blocks
    num_heads: int = 4  # Number of attention heads (must divide hidden_dim)
    dropout: float = 0.1  # Dropout rate for regularization
    
    # Tokenization (grid-based)
    grid_size: int = 19  # Grid size for state encoder (7 for partial, 19 for FourRooms full grid)
    num_tokens: int = None  # Number of tokens from state encoder (None = grid_size * grid_size)
    max_seq_len: int = 32  # Maximum sequence length for action sequences
    
    def __post_init__(self):
        """Validate configuration and compute num_tokens if needed."""
        if self.num_tokens is None:
            self.num_tokens = self.grid_size * self.grid_size
    
    def __post_init__(self):
        """Validate configuration."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})")


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Data
    batch_size: int = 16  # Batch size for training
    data_dir: str = "data"  # Directory containing train/test pickle files
    env_name: str = "FourRooms"  # Environment name (determines dataset to load)
    
    # Optimization
    learning_rate: float = 5e-5  # Learning rate for AdamW optimizer
    weight_decay: float = 1e-5  # Weight decay for L2 regularization (AdamW)
    num_epochs: int = 100  # Number of training epochs
    
    # Diffusion
    num_diffusion_steps: int = 1000  # Number of diffusion timesteps (for noise scheduling)
    mask_schedule: str = "cosine"  # Masking schedule: 'cosine', 'linear', or 'constant'
    
    # Curriculum learning for masking (gradually increase difficulty)
    use_mask_curriculum: bool = True  # Enable curriculum learning for mask ratio
    mask_curriculum_schedule: str = "linear"  # 'linear' or 'cosine' progression
    min_mask_ratio: float = 0.1  # Starting mask ratio (10% - easier at start)
    max_mask_ratio: float = 0.75  # Ending mask ratio (75% - harder at end)
    mask_curriculum_warmup_epochs: Optional[int] = None  # None = use num_epochs
    
    # Learning rate scheduling
    lr_patience: int = 5  # Patience for LR scheduler (epochs before reducing LR)
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints"  # Directory to save model checkpoints
    
    # Pretraining (Hybrid Training Approach)
    pretrained_state_encoder_path: Optional[str] = None  # Path to pretrained state encoder checkpoint
    freeze_state_encoder: bool = False  # Whether to freeze state encoder during training (Stage 2)
    
    # Logging
    use_wandb: bool = False  # Enable Weights & Biases logging


@dataclass
class MCTDConfig:
    """MCTD search configuration."""
    # Search parameters
    num_simulations: int = 50  # Number of MCTD simulations per search
    exploration_const: float = 1.414  # UCT exploration constant (c)
    
    # Denoising parameters
    guidance_scales: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])  # Guidance scales for expansion
    sparse_timesteps: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.2, 0.0])  # Sparse timesteps for fast rollout
    denoising_step_size: float = 0.2  # Denoising step size (Δt)
    
    # Reward parameters
    reward_alpha: float = 0.1  # Weight for cosine similarity reward term
    use_similarity_reward: bool = True  # Whether to use reference path similarity in reward


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mctd: MCTDConfig = field(default_factory=MCTDConfig)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary format (for compatibility with existing code)."""
        return {
            # Model config (for DiffusionPolicy initialization)
            'num_actions': self.model.num_actions,
            'hidden_dim': self.model.hidden_dim,
            'num_layers': self.model.num_layers,
            'num_heads': self.model.num_heads,
            'grid_size': self.model.grid_size,
            'num_tokens': self.model.num_tokens,
            'max_seq_len': self.model.max_seq_len,
            'dropout': self.model.dropout,
            
            # Training config
            'batch_size': self.training.batch_size,
            'learning_rate': self.training.learning_rate,
            'weight_decay': self.training.weight_decay,
            'num_epochs': self.training.num_epochs,
            'num_diffusion_steps': self.training.num_diffusion_steps,
            'mask_schedule': self.training.mask_schedule,
            'use_mask_curriculum': self.training.use_mask_curriculum,
            'mask_curriculum_schedule': self.training.mask_curriculum_schedule,
            'min_mask_ratio': self.training.min_mask_ratio,
            'max_mask_ratio': self.training.max_mask_ratio,
            'mask_curriculum_warmup_epochs': self.training.mask_curriculum_warmup_epochs,
            'lr_patience': self.training.lr_patience,
            'data_dir': self.training.data_dir,
            'env_name': self.training.env_name,
            'checkpoint_dir': self.training.checkpoint_dir,
            'pretrained_state_encoder_path': self.training.pretrained_state_encoder_path,
            'freeze_state_encoder': self.training.freeze_state_encoder,
            'use_wandb': self.training.use_wandb,
        }


# Default configuration instance
# Initialize with FourRooms settings (19x19 grid)
_default_model_config = ModelConfig(
    grid_size=19,  # FourRooms full grid is 19x19
    num_tokens=19 * 19,  # 361 tokens for full grid
)

default_config = ExperimentConfig(
    model=_default_model_config,
)


def get_model_config() -> ModelConfig:
    """Get default model configuration."""
    return default_config.model


def get_training_config() -> TrainingConfig:
    """Get default training configuration."""
    return default_config.training


def get_mctd_config() -> MCTDConfig:
    """Get default MCTD configuration."""
    return default_config.mctd


def get_experiment_config() -> ExperimentConfig:
    """Get complete default experiment configuration."""
    return default_config


def get_grid_size_for_env(env_name: str) -> int:
    """
    Get grid size for a given environment.
    
    Args:
        env_name: Environment name (e.g., 'FourRooms', 'Empty-8x8')
    
    Returns:
        Grid size (e.g., 19 for FourRooms full grid, 9 for Empty-8x8 full grid, 7 for partial)
    """
    env_name_lower = env_name.lower()
    
    # Determine grid size based on environment
    if 'fourrooms' in env_name_lower:
        # FourRooms: typically 19x19 grid (17x17 cells + border)
        return 19
    elif 'empty' in env_name_lower and '8x8' in env_name_lower:
        # Empty-8x8: typically 9x9 grid (8x8 cells + border)
        return 9
    elif 'empty' in env_name_lower:
        # Other Empty variants: use 7x7 (partial) or infer from size
        return 7
    else:
        # Default: assume 19x19 for complex environments, 7x7 for simple
        return 19  # Conservative: use larger size


def load_config_from_dict(config_dict: dict) -> ExperimentConfig:
    """Load configuration from dictionary (e.g., from checkpoint).
    
    Handles both old-style flat dicts and new-style nested dicts.
    """
    # Handle nested config (if 'model' and 'training' keys exist)
    if 'model' in config_dict and 'training' in config_dict:
        model_dict = config_dict['model']
        training_dict = config_dict['training']
    else:
        # Flat dict (old format or from to_dict())
        model_dict = config_dict
        training_dict = config_dict
    
    # Get grid_size from dict or infer from env_name
    grid_size = model_dict.get('grid_size')
    if grid_size is None:
        # Infer from environment name if available
        env_name = training_dict.get('env_name', 'FourRooms')
        grid_size = get_grid_size_for_env(env_name)
    
    # Get num_tokens from dict or compute from grid_size
    num_tokens = model_dict.get('num_tokens')
    if num_tokens is None:
        # Compute from grid_size
        num_tokens = grid_size * grid_size
    
    model = ModelConfig(
        num_actions=model_dict.get('num_actions', 7),
        hidden_dim=model_dict.get('hidden_dim', 128),
        num_layers=model_dict.get('num_layers', 8),
        num_heads=model_dict.get('num_heads', 4),
        grid_size=grid_size,
        num_tokens=num_tokens,
        max_seq_len=model_dict.get('max_seq_len', 64),
        dropout=model_dict.get('dropout', 0.1),
    )
    
    training = TrainingConfig(
        batch_size=training_dict.get('batch_size', 16),
        learning_rate=training_dict.get('learning_rate', 1e-4),
        weight_decay=training_dict.get('weight_decay', 1e-5),
        num_epochs=training_dict.get('num_epochs', 50),
        num_diffusion_steps=training_dict.get('num_diffusion_steps', 100),
        mask_schedule=training_dict.get('mask_schedule', 'cosine'),
        use_mask_curriculum=training_dict.get('use_mask_curriculum', True),
        mask_curriculum_schedule=training_dict.get('mask_curriculum_schedule', 'linear'),
        min_mask_ratio=training_dict.get('min_mask_ratio', 0.1),
        max_mask_ratio=training_dict.get('max_mask_ratio', 0.75),
        mask_curriculum_warmup_epochs=training_dict.get('mask_curriculum_warmup_epochs', None),
        lr_patience=training_dict.get('lr_patience', 5),
        data_dir=training_dict.get('data_dir', 'data'),
        env_name=training_dict.get('env_name', 'FourRooms'),
        checkpoint_dir=training_dict.get('checkpoint_dir', 'checkpoints'),
        pretrained_state_encoder_path=training_dict.get('pretrained_state_encoder_path', None),
        freeze_state_encoder=training_dict.get('freeze_state_encoder', False),
        use_wandb=training_dict.get('use_wandb', False),
    )
    
    return ExperimentConfig(model=model, training=training)
