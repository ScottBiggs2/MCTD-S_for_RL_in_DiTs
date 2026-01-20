"""
Dataset class for loading expert trajectories.
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import pickle
from pathlib import Path


class TrajectoryDataset(Dataset):
    """
    Dataset for loading expert trajectories.
    
    Each trajectory contains:
    - states: List of state dictionaries
    - actions: List of action indices
    - length: Trajectory length
    """
    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        max_seq_len: int = 64,
        pad_action: int = 0,  # Padding action index
    ):
        """
        Args:
            trajectories: List of trajectory dictionaries
            max_seq_len: Maximum sequence length (pad or truncate)
            pad_action: Action index to use for padding
        """
        self.trajectories = trajectories
        self.max_seq_len = max_seq_len
        self.pad_action = pad_action
    
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
        
        batch_states = {
            'grid': torch.tensor(initial_state['grid'], dtype=torch.float32),  # [H, W, 3] -> DataLoader batches to [B, H, W, 3]
            'direction': torch.tensor(initial_state['direction'], dtype=torch.long),  # [] (scalar) -> DataLoader batches to [B]
        }
        
        return {
            'states': batch_states,
            'actions': actions_tensor,
            'length': length,
        }
