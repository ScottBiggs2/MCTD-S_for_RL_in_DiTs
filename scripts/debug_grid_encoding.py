"""
Debug script to check if agent and goal positions are visible in grid encoding.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import gymnasium as gym

try:
    import minigrid
    from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
except ImportError:
    print("ERROR: minigrid not installed")
    sys.exit(1)

from src.environments.minigrid_wrapper import get_full_grid_image

# Create environment
env = gym.make('MiniGrid-FourRooms-v0')
obs, info = env.reset(seed=42)

# Get full grid image
grid_image = get_full_grid_image(env)

# Get agent and goal positions
agent_pos = env.unwrapped.agent_pos
goal_pos = None

# Find goal position by scanning grid
grid = env.unwrapped.grid
for i in range(grid.height):
    for j in range(grid.width):
        cell = grid.get(j, i)
        if cell is not None and hasattr(cell, 'type') and cell.type == 'goal':
            goal_pos = (j, i)
            break
    if goal_pos:
        break

print("=" * 60)
print("Grid Encoding Debug")
print("=" * 60)
print(f"\nGrid shape: {grid_image.shape}")
print(f"Agent position: {agent_pos}")
print(f"Goal position: {goal_pos}")

print("\n" + "=" * 60)
print("Agent Position Encoding:")
print("=" * 60)
if agent_pos:
    ax, ay = agent_pos
    agent_encoding = grid_image[ay, ax, :]
    print(f"Position ({ax}, {ay}):")
    print(f"  Channel 0 (object type): {agent_encoding[0]}")
    print(f"  Channel 1 (color): {agent_encoding[1]}")
    print(f"  Channel 2 (state/direction): {agent_encoding[2]}")
    print(f"  Expected: object_type=10 (agent marker), direction={env.unwrapped.agent_dir}")

print("\n" + "=" * 60)
print("Goal Position Encoding:")
print("=" * 60)
if goal_pos:
    gx, gy = goal_pos
    goal_encoding = grid_image[gy, gx, :]
    print(f"Position ({gx}, {gy}):")
    print(f"  Channel 0 (object type): {goal_encoding[0]}")
    print(f"  Channel 1 (color): {goal_encoding[1]}")
    print(f"  Channel 2 (state): {goal_encoding[2]}")
    print(f"  Expected: object_type={OBJECT_TO_IDX.get('goal', 8)}")
    
    # Check if goal is visible (not overwritten by agent)
    if goal_pos == tuple(agent_pos):
        print("  ⚠️  WARNING: Agent is on goal! Goal encoding may be overwritten.")
    else:
        print("  ✓ Goal is separate from agent position")
else:
    print("  ⚠️  WARNING: Goal position not found in grid!")

print("\n" + "=" * 60)
print("Object Type Distribution:")
print("=" * 60)
unique_types, counts = np.unique(grid_image[:, :, 0], return_counts=True)
for obj_type, count in zip(unique_types, counts):
    type_name = None
    for name, idx in OBJECT_TO_IDX.items():
        if idx == obj_type:
            type_name = name
            break
    if obj_type == 10:
        type_name = "AGENT_MARKER (custom)"
    print(f"  Type {obj_type} ({type_name if type_name else 'unknown'}): {count} cells")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
if goal_pos and goal_pos != tuple(agent_pos):
    goal_visible = grid_image[goal_pos[1], goal_pos[0], 0] == OBJECT_TO_IDX.get('goal', 8)
    if goal_visible:
        print("✓ Goal IS visible in grid encoding")
    else:
        print("✗ Goal is NOT properly encoded in grid")
        
agent_visible = grid_image[agent_pos[1], agent_pos[0], 0] == 10
if agent_visible:
    print("✓ Agent IS marked in grid encoding")
else:
    print("✗ Agent is NOT properly marked in grid")

env.close()
