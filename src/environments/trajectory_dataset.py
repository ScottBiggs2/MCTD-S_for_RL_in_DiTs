"""
Dataset class for loading expert trajectories.
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path
import random


class TrajectoryDataset(Dataset):
    """
    Dataset for loading expert trajectories.
    
    Each trajectory contains:
    - states: List of state dictionaries
    - actions: List of action indices
    - length: Trajectory length
    
    Supports data augmentation via grid rotations/flips.
    Actions are agent-relative, so they remain valid after transformations.
    """
    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        max_seq_len: int = 64,
        pad_action: int = 0,  # Padding action index
        use_augmentation: bool = True,
        grid_size: int = 19,  # For FourRooms, needed for augmentation
    ):
        """
        Args:
            trajectories: List of trajectory dictionaries
            max_seq_len: Maximum sequence length (pad or truncate)
            pad_action: Action index to use for padding
            use_augmentation: Whether to apply random augmentations (rotations/flips)
            grid_size: Size of grid (needed for augmentation transformations)
        """
        self.trajectories = trajectories
        self.max_seq_len = max_seq_len
        self.pad_action = pad_action
        self.use_augmentation = use_augmentation
        self.grid_size = grid_size
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        
        states = traj['states']
        actions = traj['actions']
        length = min(len(actions), self.max_seq_len)
        
        # Pad or truncate states
        states_padded = []
        for i in range(self.max_seq_len):
            if i < len(states):
                state = states[i]
            else:
                # Use last state for padding
                state = states[-1] if states else {}
            states_padded.append(state)
        
        # Pad or truncate actions
        actions_tensor = torch.tensor(actions[:self.max_seq_len], dtype=torch.long)
        if len(actions_tensor) < self.max_seq_len:
            padding = torch.full((self.max_seq_len - len(actions_tensor),), self.pad_action, dtype=torch.long)
            actions_tensor = torch.cat([actions_tensor, padding])
        
        # Convert states to batched format
        # IMPORTANT: For masked diffusion, we only condition on the INITIAL state
        # The model predicts actions from the initial state, not from each state in sequence
        # So we only return the first state (initial condition)
        # DataLoader will add batch dimension automatically
        initial_state = states_padded[0] if states_padded else states[-1] if states else {}
        
        # Extract initial state components
        initial_grid = torch.tensor(initial_state['grid'], dtype=torch.float32)
        # Handle different grid formats
        if initial_grid.dim() == 1:
            # Flattened: reshape to (H, W, 3)
            initial_grid = initial_grid.reshape(self.grid_size, self.grid_size, 3)
        elif initial_grid.dim() == 3 and initial_grid.shape[0] == 3:
            # Channels first: convert to channels last
            initial_grid = initial_grid.permute(1, 2, 0)
        
        initial_direction = torch.tensor(initial_state['direction'], dtype=torch.long)
        
        # Apply random augmentation if enabled
        # Actions are agent-relative (turn left/right/forward), so they remain valid
        # after grid transformations - we just need to transform the initial state
        # The expert action sequence remains correct because actions are relative to agent orientation
        if self.use_augmentation and random.random() < 0.75:  # 75% chance of augmentation
            from src.utils.data_augmentation import apply_grid_augmentation
            aug_types = ['rot90', 'rot180', 'rot270', 'hflip', 'vflip']
            aug_type = random.choice(aug_types)
            
            # Apply augmentation to initial state
            # Note: We don't have agent/goal positions here, so pass None
            initial_grid_aug, _, _, initial_direction_aug = apply_grid_augmentation(
                initial_grid,
                None,  # agent_pos - not needed for DiT training
                None,  # goal_pos - not needed for DiT training
                initial_direction,
                self.grid_size,
                aug_type
            )
            
            initial_grid = initial_grid_aug
            initial_direction = initial_direction_aug
        
        batch_states = {
            'grid': initial_grid,  # [H, W, 3] -> DataLoader batches to [B, H, W, 3]
            'direction': initial_direction,  # [] (scalar) -> DataLoader batches to [B]
        }
        
        return {
            'states': batch_states,
            'actions': actions_tensor,
            'length': length,
        }
