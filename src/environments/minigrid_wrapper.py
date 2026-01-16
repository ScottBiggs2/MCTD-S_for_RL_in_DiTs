"""
Wrapper for MiniGrid environments.
Provides state embeddings and branch point detection.
"""
import gymnasium as gym
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any

# Import minigrid to register environments
try:
    import minigrid
    # This registers all MiniGrid environments with gymnasium
except ImportError:
    raise ImportError(
        "minigrid package not found. Install with: pip install minigrid"
    )


class MazeEnvironment:
    """
    Wrapper for MiniGrid environments.
    Provides state embeddings and branch point detection.
    
    Note: This is a basic implementation. State representation choices
    may need refinement based on research decisions.
    """
    def __init__(self, env_name: str = "MiniGrid-Empty-8x8-v0", max_steps: int = 100):
        """
        Initialize MiniGrid environment wrapper.
        
        Args:
            env_name: Name of the MiniGrid environment
            max_steps: Maximum steps per episode
        """
        # Ensure minigrid is imported to register environments
        import minigrid
        
        try:
            self.env = gym.make(env_name)
        except (gym.error.NameNotFound, gym.error.UnregisteredEnv) as e:
            # Try alternative naming or provide helpful error
            try:
                available = [k for k in gym.envs.registry.keys() if 'MiniGrid' in k or 'minigrid' in k.lower()]
                available_str = available[:10] if len(available) > 10 else available
            except:
                available_str = "Unable to list environments"
            
            raise ValueError(
                f"Environment '{env_name}' not found. "
                f"Make sure minigrid is installed: pip install minigrid\n"
                f"Available MiniGrid environments: {available_str}\n"
                f"Original error: {e}"
            )
        
        # Get action space from environment
        self.action_space = self.env.action_space.n if hasattr(self.env.action_space, 'n') else 7
        self.max_steps = max_steps
        self.env_name = env_name
        
    def reset(self) -> Dict[str, torch.Tensor]:
        """
        Reset environment and return initial state.
        
        Returns:
            Dictionary with state representation:
            - 'grid': flattened grid observation [grid_size * grid_size * 3]
            - 'direction': agent direction [1]
            - 'position': agent position [2] if available, else None
        """
        obs, info = self.env.reset()
        return self.get_state_embedding(obs)
    
    def step(self, action: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict]:
        """
        Execute action and return (state, reward, done, info).
        
        Args:
            action: Discrete action index (0-6)
            
        Returns:
            state: State dictionary
            reward: Reward signal
            done: Whether episode terminated
            info: Additional information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        state = self.get_state_embedding(obs)
        return state, reward, done, info
    
    def get_state_embedding(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Convert MiniGrid observation to state representation.
        
        obs['image']: (7, 7, 3) partially observable view
            - Channel 0: object type
            - Channel 1: object color
            - Channel 2: object state
        obs['direction']: agent direction (0-3)
        
        Returns:
            Dictionary with:
            - 'grid': flattened grid observation [147] (7*7*3)
            - 'direction': facing direction [1]
            - 'position': (x, y) if available, else None
        """
        # Flatten grid observation: (7, 7, 3) -> (147,)
        grid_flat = obs['image'].flatten()
        direction = obs['direction']
        
        # Try to get absolute position (available in some envs)
        position = None
        if hasattr(self.env.unwrapped, 'agent_pos'):
            position = self.env.unwrapped.agent_pos
        
        state_dict = {
            'grid': torch.tensor(grid_flat, dtype=torch.float32),
            'direction': torch.tensor(direction, dtype=torch.long),
        }
        
        if position is not None:
            state_dict['position'] = torch.tensor(position, dtype=torch.long)
        
        return state_dict
    
    def get_branch_points(self, trajectory: List[Tuple[Dict, int, float]]) -> List[int]:
        """
        Identify decision points in a trajectory.
        
        Branch points are positions where:
        - Multiple valid actions exist
        - Near doorways or intersections
        - Key objects nearby
        
        Args:
            trajectory: List of (state, action, reward) tuples
            
        Returns:
            List of indices where branching occurs
        """
        branch_indices = []
        
        for i, (state, action, reward) in enumerate(trajectory):
            # Heuristic: detect grid cells with multiple open neighbors
            grid = state['grid'].reshape(7, 7, 3)
            
            # Count empty adjacent cells (simplified)
            if self._is_branch_point(grid):
                branch_indices.append(i)
        
        return branch_indices
    
    def _is_branch_point(self, grid: np.ndarray) -> bool:
        """
        Heuristic to detect if current position is a branch point.
        
        Args:
            grid: (7, 7, 3) grid observation
            
        Returns:
            True if this appears to be a branch point
        """
        # Center of observable grid (agent position)
        center = grid[3, 3]
        
        # Count empty neighbors (object type == 1 for empty)
        neighbors = [
            grid[2, 3, 0],  # forward
            grid[3, 2, 0],  # left
            grid[3, 4, 0],  # right
        ]
        
        empty_count = sum(n == 1 for n in neighbors)
        return empty_count >= 2  # Branch if 2+ directions available
    
    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()
