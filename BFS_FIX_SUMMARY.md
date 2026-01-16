# BFS Fix Summary

## Issues Fixed

### 1. **Primary Issue: Partial Observations** ✅ FIXED
**Problem**: BFS was using `obs['image']` (7x7 partial view) for state representation, making it impossible to plan effectively in complex environments like FourRooms.

**Fix**: Changed to use full grid state via `env.unwrapped.agent_pos` and `env.unwrapped.agent_dir` for state hashing and BFS planning.

**Impact**: This should dramatically reduce failure rate from ~15% to near 0%.

### 2. **State Hashing** ✅ FIXED
**Problem**: Hashing partial observations caused collisions and missed states.

**Fix**: New `hash_state()` function uses `(agent_pos, agent_dir)` tuple for accurate state representation.

### 3. **Reward Check** ✅ IMPROVED
**Problem**: Code checked `terminated and reward > 0`, but reward formula is `1 - 0.9 * (step_count / max_steps)`.

**Fix**: Check `terminated and reward > 0.1` to account for reward formula.

### 4. **Max Iterations** ✅ INCREASED
**Problem**: 10000 iterations might not be enough for complex paths.

**Fix**: Increased to 50000 for complex environments like FourRooms.

### 5. **Replay Logic** ✅ IMPROVED
**Problem**: If replay terminated early, code would continue incorrectly.

**Fix**: Added proper replay success checking to skip invalid paths.

## Key Changes

1. **New `hash_state()` function**:
   ```python
   def hash_state(env):
       pos = env.unwrapped.agent_pos
       dir = env.unwrapped.agent_dir
       return hash((tuple(pos), dir))
   ```

2. **BFS now uses full state**:
   - Queue stores `(actions, seed)` instead of `(obs, actions, ...)`
   - State hashing uses position + direction
   - No longer relies on partial observations

3. **Better error handling**:
   - Checks replay success before exploring actions
   - Skips invalid paths properly

## Expected Results

- **Empty-8x8**: Should still work perfectly (was already working)
- **FourRooms**: Failure rate should drop from ~15% to <1%
- **Performance**: Slightly slower due to more iterations, but more reliable

## Testing

Run data generation again:
```bash
python scripts/generate_expert_data.py
```

Expected:
- Empty-8x8: 500/500 successful (unchanged)
- FourRooms: ~495-500/500 successful (up from ~425/500)

## Why This Works

1. **Full State Information**: BFS now has complete information about agent position and direction, allowing it to plan optimal paths through the full maze.

2. **Accurate State Representation**: Using `(pos, dir)` tuple ensures each unique state is correctly identified, avoiding collisions.

3. **Proper Goal Detection**: Checking `reward > 0.1` accounts for MiniGrid's reward formula while still detecting successful goal completion.

## References

- MiniGrid documentation: Full state access via `env.unwrapped.agent_pos` and `env.unwrapped.agent_dir`
- BFS best practices: Always use full state representation for optimal planning
- Gymnasium API: Proper handling of `terminated` and `truncated` flags
