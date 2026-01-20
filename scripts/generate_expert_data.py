"""
Generate expert trajectories using A* or BFS solver.

Supports multiprocessing for parallel data generation.
Creates train/test split with isolated test environments.

A* is the default solver and is typically faster than BFS because it uses
a heuristic (Manhattan distance) to guide the search toward the goal.
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
from typing import List, Tuple, Dict, Any, Optional
import random
import heapq
import multiprocessing as mp
from functools import partial

# Import minigrid to register environments
try:
    import minigrid
except ImportError:
    raise ImportError("minigrid package not found. Install with: pip install minigrid")

from src.environments.minigrid_wrapper import MazeEnvironment


def hash_state(env):
    """
    Create hash of full state (position + direction) for visited set.
    
    Uses full grid state instead of partial observations for accurate pathfinding.
    """
    pos = env.unwrapped.agent_pos
    dir = env.unwrapped.agent_dir
    return hash((tuple(pos), dir))


def get_goal_position(env) -> Optional[Tuple[int, int]]:
    """
    Extract goal position from environment.
    
    Returns:
        Goal position (x, y) or None if not found
    """
    if not hasattr(env.unwrapped, 'grid'):
        return None
    
    grid = env.unwrapped.grid
    for i in range(grid.width):
        for j in range(grid.height):
            cell = grid.get(i, j)
            if cell and hasattr(cell, 'type') and cell.type == 'goal':
                return (i, j)
    return None


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Manhattan distance heuristic for A*.
    
    Args:
        pos1: (x1, y1)
        pos2: (x2, y2)
    
    Returns:
        Manhattan distance |x1-x2| + |y1-y2|
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar_solve_maze(env_name: str, max_iterations: int = 50000, seed: int = None) -> Tuple[List[int], bool]:
    """
    Use A* algorithm to find shortest path in MiniGrid environment.
    
    A* is more efficient than BFS because it uses a heuristic (Manhattan distance)
    to guide the search toward the goal, exploring promising paths first.
    
    Args:
        env_name: Name of the environment
        max_iterations: Maximum A* iterations (safety limit)
        seed: Random seed for environment
    
    Returns:
        actions: List of action indices
        success: Whether goal was reached
    """
    # Create environment and get goal position
    env = gym.make(env_name)
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    
    # Get goal position for heuristic
    goal_pos = get_goal_position(env)
    if goal_pos is None:
        # Fallback: if we can't find goal, use BFS
        env.close()
        return bfs_solve_maze(env_name, max_iterations, seed)
    
    initial_pos = tuple(env.unwrapped.agent_pos)
    initial_dir = env.unwrapped.agent_dir
    initial_hash = hash_state(env)
    
    # A* priority queue: (f_score, g_score, action_sequence, env_seed)
    # f_score = g_score + h_score (estimated total cost)
    # g_score = actual cost from start (number of steps)
    # h_score = heuristic estimate to goal (Manhattan distance)
    initial_h = manhattan_distance(initial_pos, goal_pos)
    initial_f = 0 + initial_h  # g=0 (at start), h=manhattan distance
    
    # Priority queue: (f_score, g_score, action_sequence, env_seed)
    # Use g_score as tie-breaker for determinism
    heap = [(initial_f, 0, [], seed)]
    visited = {initial_hash: 0}  # state_hash -> best_g_score
    
    iterations = 0
    
    while heap and iterations < max_iterations:
        f_score, g_score, actions, env_seed = heapq.heappop(heap)
        
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
                replay_success = False
                break
        
        if not replay_success:
            test_env.close()
            iterations += 1
            continue
        
        # Get current state
        current_pos = tuple(test_env.unwrapped.agent_pos)
        current_dir = test_env.unwrapped.agent_dir
        current_hash = hash_state(test_env)
        
        # Check if reached goal
        if current_pos == goal_pos:
            test_env.close()
            env.close()
            return actions, True
        
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
            if terminated and reward > 0.1:
                action_env.close()
                test_env.close()
                env.close()
                return actions + [action], True
            
            # Add to queue if not done
            if not (terminated or truncated):
                next_pos = tuple(action_env.unwrapped.agent_pos)
                next_hash = hash_state(action_env)
                
                # Calculate g_score (cost from start)
                next_g = g_score + 1
                
                # Skip if we've seen this state with better or equal g_score
                # In A*, if we've already found a path to a state with lower or equal cost, skip it
                if next_hash in visited and visited[next_hash] <= next_g:
                    action_env.close()
                    continue
                
                # Update best g_score for this state
                visited[next_hash] = next_g
                
                # Calculate heuristic (Manhattan distance to goal)
                next_h = manhattan_distance(next_pos, goal_pos)
                
                # f_score = g + h (A* evaluation function)
                next_f = next_g + next_h
                
                # Add to priority queue
                heapq.heappush(heap, (next_f, next_g, actions + [action], env_seed))
            
            action_env.close()
        
        test_env.close()
        iterations += 1
    
    env.close()
    return [], False


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


def collect_trajectory(env: gym.Env, actions: List[int], seed: Optional[int] = None, use_full_grid: bool = True) -> Dict[str, Any]:
    """
    Collect full trajectory by executing actions.
    
    Args:
        env: Environment (should already be reset, but we reset again to be safe)
        actions: List of actions to execute
        seed: Optional seed for environment reset (must match BFS seed!)
        use_full_grid: If True, use full grid observations; if False, use partial 7x7 view
    
    Returns:
        Dictionary with states, actions, and metadata
    """
    # Import here to avoid circular imports
    from src.environments.minigrid_wrapper import get_full_grid_image
    
    # CRITICAL: Reset with the SAME seed that was used for BFS solving
    # If we don't use the seed, we'll get a different environment state!
    # Cast to int for gymnasium compatibility
    if seed is not None:
        obs, _ = env.reset(seed=int(seed))
    else:
        obs, _ = env.reset()
    
    trajectory = []
    reached_goal = False
    
    # IMPORTANT: We collect state-action pairs where state is BEFORE the action
    # So trajectory[i] = (state at time i, action[i])
    # This means the initial state (t=0) is included as trajectory[0]['state']
    
    for action in actions:
        # Get grid image (full or partial)
        if use_full_grid:
            # Render full grid from environment
            grid_image = get_full_grid_image(env)
        else:
            # Use partial observation from obs['image']
            grid_image = obs['image']
        
        state = {
            'grid': grid_image.flatten(),
            'direction': obs['direction'],
        }
        
        trajectory.append({
            'state': state,
            'action': action,
        })
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if we reached the goal (successful termination)
        if terminated and reward > 0.1:
            reached_goal = True
        
        if terminated or truncated:
            break
    
    # IMPORTANT: Add "done" token (action 6) at the end of successful trajectories
    # This teaches the model when to stop generating actions
    actions_list = [t['action'] for t in trajectory]
    if reached_goal:
        # Add done token with final state (state at goal)
        # Get final state after all actions
        if use_full_grid:
            final_grid = get_full_grid_image(env)
        else:
            final_grid = obs['image']
        
        final_state = {
            'grid': final_grid.flatten(),
            'direction': obs['direction'],
        }
        trajectory.append({
            'state': final_state,
            'action': 6,  # done token
        })
        actions_list.append(6)
    
    return {
        'states': [t['state'] for t in trajectory],
        'actions': actions_list,  # Includes done token if successful
        'length': len(actions_list),
        'success': reached_goal,
        'episode_seed': seed,  # Store seed for later use
    }


def _generate_trajectory_wrapper(args: Tuple[str, int, int, str, bool]) -> Optional[Dict[str, Any]]:
    """
    Wrapper function for multiprocessing that unpacks tuple arguments.
    
    This is a module-level function (not nested) so it can be pickled
    for multiprocessing.
    
    Args:
        args: Tuple of (env_name, episode, base_seed, solver, use_full_grid)
    
    Returns:
        Trajectory dictionary or None if failed
    """
    env_name, episode, base_seed, solver, use_full_grid = args
    return generate_single_trajectory(env_name, episode, base_seed, solver, use_full_grid=use_full_grid)


def generate_single_trajectory(
    env_name: str,
    episode: int,
    base_seed: int,
    solver: str = "astar",
    use_full_grid: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Generate a single expert trajectory (for multiprocessing).
    
    Args:
        env_name: Name of MiniGrid environment
        episode: Episode number (for seed calculation)
        base_seed: Base seed for reproducibility
        solver: Pathfinding algorithm ('astar' or 'bfs')
        use_full_grid: If True, use full grid observations; if False, use partial 7x7 view
    
    Returns:
        Trajectory dictionary or None if failed
    """
    episode_seed = base_seed + episode if base_seed is not None else None
    
    # Solve with chosen algorithm
    if solver == "astar":
        actions, success = astar_solve_maze(env_name, seed=episode_seed)
    else:
        actions, success = bfs_solve_maze(env_name, seed=episode_seed)
    
    if not success:
        return None
    
    # Collect full trajectory
    env = gym.make(env_name)
    trajectory = collect_trajectory(env, actions, seed=episode_seed, use_full_grid=use_full_grid)
    env.close()
    
    return trajectory


def generate_expert_dataset(
    env_name: str,
    num_episodes: int = 500,
    train_ratio: float = 0.8,
    output_dir: str = "data",
    seed: int = 42,
    solver: str = "astar",
    num_workers: int = 0,
    use_full_grid: bool = True,
):
    """
    Generate dataset of expert trajectories using A* or BFS.
    
    Supports multiprocessing for parallel generation.
    
    Args:
        env_name: Name of MiniGrid environment
        num_episodes: Total number of episodes to generate
        train_ratio: Ratio of episodes for training
        output_dir: Directory to save data
        seed: Random seed for reproducibility
        solver: Pathfinding algorithm ('astar' or 'bfs')
        num_workers: Number of parallel workers (0 = auto-detect CPU count, 1 = no multiprocessing)
        use_full_grid: If True, use full grid observations; if False, use partial 7x7 view
    
    Returns:
        train_trajectories, test_trajectories
    """
    random.seed(seed)
    np.random.seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_episodes} expert trajectories for {env_name}")
    print(f"Using {solver.upper()} solver...")
    print(f"Observation mode: {'FULL GRID' if use_full_grid else 'PARTIAL (7x7)'}")
    
    if num_workers <= 0:
        # Auto-detect: use all CPU cores
        num_workers = mp.cpu_count()
        print(f"Using {num_workers} parallel workers (auto-detected CPU count)")
    elif num_workers == 1:
        print("Using single process (no multiprocessing)")
    else:
        print(f"Using {num_workers} parallel workers")
    
    all_trajectories = []
    
    # Generate trajectories
    if num_workers > 1:
        # Multiprocessing approach
        # Prepare arguments as tuples: (env_name, episode, base_seed, solver, use_full_grid)
        args_list = [
            (env_name, episode, seed, solver, use_full_grid)
            for episode in range(num_episodes)
        ]
        
        with mp.Pool(processes=num_workers) as pool:
            # Use imap_unordered for streaming results (faster than imap)
            # This allows tqdm to update progress in real-time as workers complete tasks
            results = list(tqdm(
                pool.imap_unordered(_generate_trajectory_wrapper, args_list),
                total=num_episodes,
                desc="Generating trajectories",
                unit="traj"
            ))
            
            # Filter out None results (failed trajectories)
            all_trajectories = [r for r in results if r is not None]
    else:
        # Single process approach (original)
        for episode in tqdm(range(num_episodes), desc="Generating trajectories"):
            trajectory = generate_single_trajectory(
                env_name=env_name,
                episode=episode,
                base_seed=seed,
                solver=solver,
                use_full_grid=use_full_grid
            )
            
            if trajectory is None:
                print(f"Warning: Episode {episode} failed to solve")
                continue
            
            all_trajectories.append(trajectory)
    
    successful = len(all_trajectories)
    print(f"Successfully generated {successful}/{num_episodes} trajectories")
    if successful > 0:
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate expert trajectories using A* or BFS")
    parser.add_argument('--env', type=str, default=None,
                       help='Environment name (e.g., MiniGrid-FourRooms-v0). If not provided, generates for all configured envs.')
    parser.add_argument('--num_episodes', type=int, default=500,
                       help='Number of episodes to generate per environment')
    parser.add_argument('--solver', type=str, default='astar', choices=['astar', 'bfs'],
                       help='Pathfinding algorithm: astar (faster) or bfs (slower but simpler)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of parallel workers (0 or negative = auto-detect CPU count, 1 = no multiprocessing, >1 = use N workers)')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for pickle files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of episodes for training (rest for testing)')
    parser.add_argument('--use_full_grid', action='store_true', default=True,
                       help='Use full grid observations instead of partial 7x7 view (default: True)')
    parser.add_argument('--use_partial_grid', action='store_false', dest='use_full_grid',
                       help='Use partial 7x7 view instead of full grid')
    
    args = parser.parse_args()
    
    # Determine environments to process
    if args.env:
        envs = [(args.env, args.num_episodes)]
    else:
        # Default: generate for configured environments
        envs = [
            ("MiniGrid-Empty-8x8-v0", args.num_episodes),
            ("MiniGrid-FourRooms-v0", args.num_episodes),
        ]
    
    for env_name, num_episodes in envs:
        print(f"\n{'='*60}")
        print(f"Processing {env_name}")
        print(f"{'='*60}")
        
        try:
            train_data, test_data = generate_expert_dataset(
                env_name=env_name,
                num_episodes=num_episodes,
                train_ratio=args.train_ratio,
                output_dir=args.output_dir,
                seed=args.seed,
                solver=args.solver,
                num_workers=args.num_workers,
                use_full_grid=args.use_full_grid,
            )
        except Exception as e:
            print(f"Error generating data for {env_name}: {e}")
            import traceback
            traceback.print_exception(*sys.exc_info())
            continue
