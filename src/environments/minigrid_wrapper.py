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


def get_full_grid_image(env) -> np.ndarray:
    """
    Render full grid as RGB image (same format as obs['image']).
    
    Uses MiniGrid's grid rendering to create a full-board observation.
    This allows the model to see the entire maze, not just a partial view.
    
    Args:
        env: Gymnasium MiniGrid environment (can be wrapped)
        
    Returns:
        full_image: (H, W, 3) numpy array with full grid observation
                    Same format as obs['image'] but covering entire grid
    """
    unwrapped = env.unwrapped
    
    # Get grid dimensions
    grid = unwrapped.grid
    width = grid.width
    height = grid.height
    
    # Initialize image array (H, W, 3)
    # Channel 0: object type
    # Channel 1: object color  
    # Channel 2: object state
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get agent position and direction for rendering
    agent_pos = unwrapped.agent_pos
    agent_dir = unwrapped.agent_dir
    
    # Use MiniGrid's object type constants for correct encoding
    # Import here to avoid issues if minigrid not available
    try:
        from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
    except ImportError:
        # Fallback if constants not available
        OBJECT_TO_IDX = {'empty': 1, 'wall': 2, 'door': 4, 'key': 5, 'ball': 6, 'goal': 8}
        COLOR_TO_IDX = {'red': 0, 'green': 1, 'blue': 2, 'purple': 3, 'yellow': 4, 'grey': 5}
        STATE_TO_IDX = {}
    
    # Render each cell in the grid
    for i in range(height):
        for j in range(width):
            cell = grid.get(j, i)  # Note: grid.get(x, y) uses (x, y) = (j, i)
            
            # Start with empty encoding
            obj_type_idx = OBJECT_TO_IDX.get('empty', 1)
            color_idx = 0
            state_idx = 0
            
            if cell is not None and hasattr(cell, 'type'):
                # Get object type encoding
                obj_type = cell.type
                if obj_type in OBJECT_TO_IDX:
                    obj_type_idx = OBJECT_TO_IDX[obj_type]
                elif isinstance(obj_type, int):
                    obj_type_idx = obj_type  # Already an integer
                
                # Get color encoding
                if hasattr(cell, 'color') and cell.color in COLOR_TO_IDX:
                    color_idx = COLOR_TO_IDX[cell.color]
                
                # Get state encoding (for doors: open/closed)
                if hasattr(cell, 'is_open'):
                    if cell.is_open:
                        state_idx = STATE_TO_IDX.get('open', 1)
                    else:
                        state_idx = STATE_TO_IDX.get('closed', 0)
                elif hasattr(cell, 'state'):
                    if cell.state in STATE_TO_IDX:
                        state_idx = STATE_TO_IDX[cell.state]
                    elif isinstance(cell.state, int):
                        state_idx = cell.state  # Already an integer
            
            # Store in image array
            image[i, j, 0] = obj_type_idx
            image[i, j, 1] = color_idx
            image[i, j, 2] = state_idx
            
            # Mark agent position (override cell if agent is here)
            # IMPORTANT ENCODING SCHEME:
            # - Channel 0 (object type): encodes object type, with special handling for agent
            #   - Goal: 8
            #   - Agent (not on goal): 10
            #   - Agent on goal: 11 (preserves goal information)
            # - Channel 1 (color): preserved from underlying cell
            # - Channel 2 (state): agent direction when agent present, otherwise cell state
            #
            # This ensures agent and goal are distinguishable even when normalized:
            # - Goal: 8/11 = 0.727
            # - Agent: 10/11 = 0.909
            # - Agent+Goal: 11/11 = 1.0
            if agent_pos is not None and i == agent_pos[1] and j == agent_pos[0]:
                # Check if this cell is a goal - if so, we need to preserve that info
                is_goal = (obj_type_idx == OBJECT_TO_IDX.get('goal', 8))
                
                if is_goal:
                    # Agent on goal: encode as special value that indicates both
                    # Use 11 (one more than max normal object type 10) to indicate agent+goal
                    # This makes it clearly distinct: goal=8 (0.727), agent=10 (0.909), agent+goal=11 (1.0)
                    image[i, j, 0] = 11  # Special marker for agent+goal
                else:
                    # Agent on non-goal: use standard agent encoding
                    image[i, j, 0] = 10  # Agent marker
                
                # Store direction in state channel (channel 2)
                # This preserves agent orientation information
                image[i, j, 2] = agent_dir
    
    # Convert to float32 and normalize (matching obs['image'] format)
    # MiniGrid observations use uint8, but we'll keep as uint8 for consistency
    return image.astype(np.uint8)


class MazeEnvironment:
    """
    Wrapper for MiniGrid environments.
    Provides state embeddings and branch point detection.
    
    Supports both partial (7x7) and full grid observations.
    Full grid observations allow the model to see the entire maze.
    """
    def __init__(
        self, 
        env_name: str = "MiniGrid-Empty-8x8-v0", 
        max_steps: int = 100,
        use_full_grid: bool = True,
    ):
        """
        Initialize MiniGrid environment wrapper.
        
        Args:
            env_name: Name of the MiniGrid environment
            max_steps: Maximum steps per episode
            use_full_grid: If True, use full grid observations; if False, use partial 7x7 view
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
        self.use_full_grid = use_full_grid
        
        # Get grid dimensions (for full grid mode)
        if use_full_grid and hasattr(self.env.unwrapped, 'grid'):
            grid = self.env.unwrapped.grid
            self.grid_width = grid.width
            self.grid_height = grid.height
        else:
            # Partial view is always 7x7
            self.grid_width = 7
            self.grid_height = 7
        
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
        
        If use_full_grid=True, renders full grid board instead of partial view.
        This allows the model to see the entire maze (goal, walls, doors, etc.).
        
        Args:
            obs: Observation dict from environment
                - obs['image']: (7, 7, 3) partial view OR (H, W, 3) full grid
                - obs['direction']: agent direction (0-3)
        
        Returns:
            Dictionary with:
            - 'grid': flattened grid observation [H*W*3]
            - 'direction': facing direction [1]
            - 'position': (x, y) if available, else None
        """
        # Get grid image (full or partial)
        if self.use_full_grid:
            # Render full grid from environment
            grid_image = get_full_grid_image(self.env)
        else:
            # Use partial observation from obs['image']
            grid_image = obs['image']
        
        # Flatten grid observation: (H, W, 3) -> (H*W*3,)
        grid_flat = grid_image.flatten()
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
