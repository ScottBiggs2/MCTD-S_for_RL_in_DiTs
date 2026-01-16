"""
Generate expert trajectories using BFS solver.

Creates train/test split with isolated test environments.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import gymnasium as gym
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import deque
from typing import List, Tuple, Dict, Any
import random

# Import minigrid to register environments
try:
    import minigrid
except ImportError:
    raise ImportError("minigrid package not found. Install with: pip install minigrid")

from src.environments.minigrid_wrapper import MazeEnvironment


def hash_state(env):
    """
    Create hash of full state (position + direction) for visited set.
    
    Uses full grid state instead of partial observations for accurate BFS.
    """
    pos = env.unwrapped.agent_pos
    dir = env.unwrapped.agent_dir
    return hash((tuple(pos), dir))


def bfs_solve_maze(env_name: str, max_iterations: int = 50000, seed: int = None) -> Tuple[List[int], bool]:
    """
    Use BFS to find shortest path in MiniGrid environment.
    
    Uses full grid state (position + direction) instead of partial observations
    for accurate planning, especially in complex environments like FourRooms.
    
    Args:
        env_name: Name of the environment
        max_iterations: Maximum BFS iterations (increased for complex envs)
        seed: Random seed for environment
    
    Returns:
        actions: List of action indices
        success: Whether goal was reached
    """
    # Create environment
    env = gym.make(env_name)
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    
    initial_hash = hash_state(env)
    
    # BFS queue: (action_sequence, env_seed)
    # We use full state (pos + dir) for hashing, not partial observations
    queue = deque([([], seed)])
    visited = {initial_hash}
    
    iterations = 0
    
    while queue and iterations < max_iterations:
        actions, env_seed = queue.popleft()
        
        # Create environment and replay to current state
        test_env = gym.make(env_name)
        if env_seed is not None:
            test_obs, _ = test_env.reset(seed=env_seed)
        else:
            test_obs, _ = test_env.reset()
        
        # Replay actions to current state
        replay_success = True
        for a in actions:
            test_obs, _, terminated, truncated, _ = test_env.step(a)
            if terminated or truncated:
                # If we terminated during replay, this path is invalid
                replay_success = False
                break
        
        if not replay_success:
            test_env.close()
            iterations += 1
            continue
        
        # Get current state hash
        current_hash = hash_state(test_env)
        
        # Try all actions
        for action in range(test_env.action_space.n):
            # Create fresh env for this action exploration
            action_env = gym.make(env_name)
            if env_seed is not None:
                action_obs, _ = action_env.reset(seed=env_seed)
            else:
                action_obs, _ = action_env.reset()
            
            # Replay to current state
            replay_ok = True
            for a in actions:
                action_obs, _, term, trun, _ = action_env.step(a)
                if term or trun:
                    replay_ok = False
                    break
            
            if not replay_ok:
                action_env.close()
                continue
            
            # Try new action
            next_obs, reward, terminated, truncated, info = action_env.step(action)
            
            # Check if reached goal
            # MiniGrid sets terminated=True and reward > 0 when goal reached
            if terminated:
                # Reward formula: 1 - 0.9 * (step_count / max_steps)
                # So reward > 0.1 indicates success
                if reward > 0.1:
                    action_env.close()
                    test_env.close()
                    env.close()
                    return actions + [action], True
            
            # Add to queue if not done and not visited
            if not (terminated or truncated):
                next_hash = hash_state(action_env)
                if next_hash not in visited:
                    visited.add(next_hash)
                    queue.append((actions + [action], env_seed))
            
            action_env.close()
        
        test_env.close()
        iterations += 1
    
    env.close()
    return [], False


def collect_trajectory(env: gym.Env, actions: List[int]) -> Dict[str, Any]:
    """
    Collect full trajectory by executing actions.
    
    Args:
        env: Environment
        actions: List of actions to execute
    
    Returns:
        Dictionary with states, actions, and metadata
    """
    obs, _ = env.reset()
    trajectory = []
    
    for action in actions:
        state = {
            'grid': obs['image'].flatten(),
            'direction': obs['direction'],
        }
        
        trajectory.append({
            'state': state,
            'action': action,
        })
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    return {
        'states': [t['state'] for t in trajectory],
        'actions': [t['action'] for t in trajectory],
        'length': len(trajectory),
        'success': True,
    }


def generate_expert_dataset(
    env_name: str,
    num_episodes: int = 500,
    train_ratio: float = 0.8,
    output_dir: str = "data",
    seed: int = 42,
):
    """
    Generate dataset of expert trajectories using BFS.
    
    Creates train/test split by randomly assigning episodes.
    For a truly isolated test set, you may want to use different
    environment seeds or generate test episodes separately.
    
    Args:
        env_name: Name of MiniGrid environment
        num_episodes: Total number of episodes to generate
        train_ratio: Ratio of episodes for training
        output_dir: Directory to save data
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_episodes} expert trajectories for {env_name}")
    print(f"Using BFS solver...")
    
    all_trajectories = []
    successful = 0
    
    # Generate trajectories
    for episode in tqdm(range(num_episodes), desc="Generating trajectories"):
        # Solve with BFS (creates its own environment)
        episode_seed = seed + episode if seed is not None else None
        actions, success = bfs_solve_maze(env_name, seed=episode_seed)
        
        if not success:
            print(f"Warning: Episode {episode} failed to solve")
            continue
        
        successful += 1
        
        # Collect full trajectory (create new env for collection)
        env = gym.make(env_name)
        if episode_seed is not None:
            env.reset(seed=episode_seed)
        else:
            env.reset()
        
        trajectory = collect_trajectory(env, actions)
        all_trajectories.append(trajectory)
        env.close()
    
    print(f"Successfully generated {successful}/{num_episodes} trajectories")
    print(f"Average trajectory length: {np.mean([t['length'] for t in all_trajectories]):.1f}")
    
    # Split into train/test
    random.shuffle(all_trajectories)
    split_idx = int(len(all_trajectories) * train_ratio)
    
    train_trajectories = all_trajectories[:split_idx]
    test_trajectories = all_trajectories[split_idx:]
    
    print(f"Train: {len(train_trajectories)} trajectories")
    print(f"Test: {len(test_trajectories)} trajectories")
    
    # Save datasets
    env_short = env_name.replace("MiniGrid-", "").replace("-v0", "")
    
    train_path = output_path / f"{env_short}_train.pkl"
    test_path = output_path / f"{env_short}_test.pkl"
    
    with open(train_path, 'wb') as f:
        pickle.dump(train_trajectories, f)
    
    with open(test_path, 'wb') as f:
        pickle.dump(test_trajectories, f)
    
    print(f"Saved train data to {train_path}")
    print(f"Saved test data to {test_path}")
    
    return train_trajectories, test_trajectories


if __name__ == "__main__":
    # Generate data for multiple environments
    envs = [
        ("MiniGrid-Empty-8x8-v0", 500),
        ("MiniGrid-FourRooms-v0", 500),
    ]
    
    for env_name, num_episodes in envs:
        print(f"\n{'='*60}")
        print(f"Processing {env_name}")
        print(f"{'='*60}")
        
        try:
            train_data, test_data = generate_expert_dataset(
                env_name=env_name,
                num_episodes=num_episodes,
                train_ratio=0.8,
                output_dir="data",
                seed=42,
            )
        except Exception as e:
            print(f"Error generating data for {env_name}: {e}")
            import traceback
            traceback.print_exception(*sys.exc_info())
            continue
