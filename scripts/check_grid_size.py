"""Check grid sizes for MiniGrid environments."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gymnasium as gym
import minigrid

envs = ['MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0']

for env_name in envs:
    env = gym.make(env_name)
    obs, info = env.reset()
    
    print(f"\n{env_name}:")
    print(f"  Partial obs['image'] shape: {obs['image'].shape}")  # Should be (7, 7, 3)
    
    # Get full grid info
    grid = env.unwrapped.grid
    print(f"  Full grid size: {grid.width} x {grid.height}")
    
    # Try to get full grid image
    # MiniGrid has a method to render the full grid
    if hasattr(env.unwrapped, 'render'):
        full_img = env.unwrapped.render()  # Returns full grid RGB array
        if full_img is not None:
            print(f"  Full render shape: {full_img.shape if hasattr(full_img, 'shape') else 'N/A'}")
    
    # Check if we can access full grid as image
    # The grid object has methods to convert to image
    try:
        # MiniGrid can render full grid using env.grid.render()
        # But we need the agent position for proper rendering
        agent_pos = env.unwrapped.agent_pos
        agent_dir = env.unwrapped.agent_dir
        
        # Try different ways to get full grid image
        if hasattr(env.unwrapped, 'gen_obs'):
            # Some MiniGrid envs have gen_obs that can generate full obs
            pass
            
        print(f"  Agent position: {agent_pos}")
        print(f"  Agent direction: {agent_dir}")
    except Exception as e:
        print(f"  Could not get agent info: {e}")
    
    env.close()
