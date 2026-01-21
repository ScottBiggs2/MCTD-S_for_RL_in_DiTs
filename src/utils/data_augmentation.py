"""
Data augmentation utilities for grid-based environments.

Supports rotations and reflections of grid observations with consistent
transformation of positions and directions.
"""
import torch
from typing import Optional, Tuple


def apply_grid_augmentation(
    grid: torch.Tensor,
    agent_pos: Optional[torch.Tensor],
    goal_pos: Optional[torch.Tensor],
    direction: torch.Tensor,
    grid_size: int,
    aug_type: str
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
    """
    Apply geometric augmentation to grid and corresponding positions/direction.
    
    IMPORTANT: All transformations are applied consistently:
    - Grid is transformed spatially
    - Positions (x, y) are transformed to match the new grid orientation
    - Direction is transformed to match the new grid orientation
    - This ensures that when the model predicts from the augmented grid,
      the labels (agent_pos, goal_pos, direction) are also in the augmented
      coordinate system, so losses are computed correctly.
    
    MiniGrid direction encoding:
    - 0 = right (east)
    - 1 = down (south)
    - 2 = left (west)
    - 3 = up (north)
    
    MiniGrid actions are agent-relative:
    - Actions (turn left/right/forward) remain valid after transformations
    - because they're relative to agent orientation, not world coordinates
    
    Args:
        grid: (H, W, 3) tensor
        agent_pos: (2,) tensor with (x, y) or None
        goal_pos: (2,) tensor with (x, y) or None
        direction: scalar tensor (0-3: right, down, left, up)
        grid_size: Size of grid (H = W = grid_size)
        aug_type: One of 'none', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip'
    
    Returns:
        (augmented_grid, augmented_agent_pos, augmented_goal_pos, augmented_direction)
        All values are in the augmented coordinate system.
    """
    H, W = grid_size, grid_size
    
    if aug_type == 'none':
        return grid, agent_pos, goal_pos, direction
    
    # Convert grid to channels-first for transforms: (H, W, 3) -> (3, H, W)
    grid_chw = grid.permute(2, 0, 1)  # (3, H, W)
    
    # Transform grid
    if aug_type == 'rot90':
        # 90° clockwise rotation
        grid_chw = torch.rot90(grid_chw, k=-1, dims=(1, 2))  # (3, H, W)
        # Transform positions: (x, y) -> (H-1-y, x)
        if agent_pos is not None:
            x, y = agent_pos[0].item(), agent_pos[1].item()
            agent_pos = torch.tensor([H - 1 - y, x], dtype=agent_pos.dtype)
        if goal_pos is not None:
            x, y = goal_pos[0].item(), goal_pos[1].item()
            goal_pos = torch.tensor([H - 1 - y, x], dtype=goal_pos.dtype)
        # Transform direction: +1 mod 4
        direction = (direction + 1) % 4
    
    elif aug_type == 'rot180':
        # 180° rotation
        grid_chw = torch.rot90(grid_chw, k=-2, dims=(1, 2))  # (3, H, W)
        # Transform positions: (x, y) -> (H-1-x, W-1-y)
        if agent_pos is not None:
            x, y = agent_pos[0].item(), agent_pos[1].item()
            agent_pos = torch.tensor([H - 1 - x, W - 1 - y], dtype=agent_pos.dtype)
        if goal_pos is not None:
            x, y = goal_pos[0].item(), goal_pos[1].item()
            goal_pos = torch.tensor([H - 1 - x, W - 1 - y], dtype=goal_pos.dtype)
        # Transform direction: +2 mod 4
        direction = (direction + 2) % 4
    
    elif aug_type == 'rot270':
        # 270° clockwise (90° counter-clockwise)
        grid_chw = torch.rot90(grid_chw, k=1, dims=(1, 2))  # (3, H, W)
        # Transform positions: (x, y) -> (y, W-1-x)
        if agent_pos is not None:
            x, y = agent_pos[0].item(), agent_pos[1].item()
            agent_pos = torch.tensor([y, W - 1 - x], dtype=agent_pos.dtype)
        if goal_pos is not None:
            x, y = goal_pos[0].item(), goal_pos[1].item()
            goal_pos = torch.tensor([y, W - 1 - x], dtype=goal_pos.dtype)
        # Transform direction: +3 mod 4
        direction = (direction + 3) % 4
    
    elif aug_type == 'hflip':
        # Horizontal flip (left-right)
        grid_chw = torch.flip(grid_chw, dims=[2])  # Flip along width dimension
        # Transform positions: (x, y) -> (W-1-x, y)
        if agent_pos is not None:
            x, y = agent_pos[0].item(), agent_pos[1].item()
            agent_pos = torch.tensor([W - 1 - x, y], dtype=agent_pos.dtype)
        if goal_pos is not None:
            x, y = goal_pos[0].item(), goal_pos[1].item()
            goal_pos = torch.tensor([W - 1 - x, y], dtype=goal_pos.dtype)
        # Transform direction: 0↔2 (right↔left), 1 and 3 stay same
        direction_map = {0: 2, 1: 1, 2: 0, 3: 3}
        direction = torch.tensor(direction_map[direction.item()], dtype=direction.dtype)
    
    elif aug_type == 'vflip':
        # Vertical flip (up-down)
        grid_chw = torch.flip(grid_chw, dims=[1])  # Flip along height dimension
        # Transform positions: (x, y) -> (x, H-1-y)
        if agent_pos is not None:
            x, y = agent_pos[0].item(), agent_pos[1].item()
            agent_pos = torch.tensor([x, H - 1 - y], dtype=agent_pos.dtype)
        if goal_pos is not None:
            x, y = goal_pos[0].item(), goal_pos[1].item()
            goal_pos = torch.tensor([x, H - 1 - y], dtype=goal_pos.dtype)
        # Transform direction: 1↔3 (down↔up), 0 and 2 stay same
        direction_map = {0: 0, 1: 3, 2: 2, 3: 1}
        direction = torch.tensor(direction_map[direction.item()], dtype=direction.dtype)
    
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    # Convert back to channels-last: (3, H, W) -> (H, W, 3)
    grid_aug = grid_chw.permute(1, 2, 0)  # (H, W, 3)
    
    return grid_aug, agent_pos, goal_pos, direction
