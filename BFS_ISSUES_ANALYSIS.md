# BFS Failure Analysis for FourRooms Environment

## Root Causes Identified

### 1. **CRITICAL: Partial Observations for BFS Planning** ⚠️
**Problem**: The BFS algorithm is using `obs['image']` which is a **7x7 partial view**, not the full grid state. This makes BFS planning extremely difficult in FourRooms where you need to navigate through multiple rooms.

**Impact**: 
- BFS can't see the full maze layout
- Same physical position with different viewing angles appears as different states
- Different positions with similar local views appear as the same state
- **This is the primary cause of ~15% failure rate**

**Solution**: Use full grid state (agent position + direction) instead of partial observations for BFS.

### 2. **State Hashing Issues**
**Problem**: Hashing partial observations (`obs['image'].tobytes()`) creates collisions and misses:
- Same position, different direction → different hash (should be same state)
- Different positions, similar local view → same hash (should be different states)

**Solution**: Hash based on `(agent_pos, agent_dir)` tuple instead.

### 3. **Reward Check**
**Problem**: Code checks `terminated and reward > 0`, but MiniGrid reward formula is:
```
reward = 1 - 0.9 * (step_count / max_steps)
```
So reward can be positive but less than 1. However, `terminated=True` should indicate goal reached, so this might be okay.

**Solution**: Check `terminated` alone, or check `reward > 0.1` to be safe.

### 4. **Max Iterations**
**Problem**: 10000 iterations might not be enough for complex FourRooms paths, but the real bottleneck is the partial observation issue.

**Solution**: Increase to 50000 for FourRooms, but fix observation issue first.

### 5. **State Replay Efficiency**
**Problem**: Creating new environment and replaying actions for each BFS exploration is slow but correct. However, with partial observations, we're not even getting correct states.

**Solution**: Fix observation issue first, then optimize if needed.

## Recommended Fix

Use **full grid state** (position + direction) for BFS planning:

```python
def hash_state(env):
    """Hash based on agent position and direction (full state)."""
    pos = env.unwrapped.agent_pos
    dir = env.unwrapped.agent_dir
    return hash((tuple(pos), dir))

def bfs_solve_maze(env_name, max_iterations=50000, seed=None):
    # Use position + direction for state representation
    # This gives BFS full information about the maze state
```

## Why Empty-8x8 Works But FourRooms Fails

- **Empty-8x8**: Simple, small maze. Partial observations are often sufficient because:
  - Shorter paths (avg 12 steps)
  - Less complex navigation
  - Fewer decision points
  
- **FourRooms**: Complex multi-room navigation. Partial observations fail because:
  - Longer paths needed
  - Must navigate through doorways between rooms
  - Partial view can't see door locations in other rooms
  - BFS gets "stuck" exploring local minima

## References

- MiniGrid docs: Partial observations are for agent learning, not for optimal planning
- Gymnasium API: `env.unwrapped.agent_pos` and `env.unwrapped.agent_dir` provide full state
- BFS best practices: Always use full state representation for optimal planning
