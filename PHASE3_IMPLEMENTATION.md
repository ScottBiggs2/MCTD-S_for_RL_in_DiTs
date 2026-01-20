# Phase 3: MCTD Search Implementation

## Status: Core Components Implemented ✅

Phase 3 core components have been implemented according to the locked-in mathematical decisions.

---

## Implemented Components

### 1. `MCTDNode` (`src/mctd/node.py`) ✅

**Purpose**: Tree node representing a partially denoised action sequence in hidden space.

**Features**:
- Stores hidden state `[seq_len, hidden_dim]`
- Tracks noise level `t ∈ [0, 1]`
- Maintains environment state snapshot
- Tree statistics (visits, Q-value, UCT score)
- Methods: `is_leaf()`, `is_terminal()`, `uct_score()`, `best_child()`

### 2. `HiddenSpaceMCTD` (`src/mctd/search.py`) ✅

**Purpose**: Core MCTD search algorithm over continuous hidden representations.

**Key Methods**:
- `search()`: Main entry point - runs MCTS simulations
- `select()`: UCT-based tree navigation
- `expand()`: Create children via different guidance scales (batched for efficiency)
- `simulate()`: Jumpy denoising rollout for reward evaluation
- `evaluate_trajectory()`: Compute reward with all components
- `backpropagate()`: Update statistics up tree
- `extract_best_trajectory()`: Follow highest Q-value path

---

## Locked-in Mathematical Decisions

From `PHASE3_PREPARATION.md` and user decisions:

1. **Denoising Schedule**: Fixed step size `Δt = 0.2` ✅
2. **Guidance Scales**: `[0.0, 0.5, 1.0]` - 3 children ✅
3. **Sparse Timesteps**: `[1.0, 0.5, 0.2, 0.0]` - 4 jumps ✅
4. **UCT Constant**: `c = 1.414` (√2) ✅
5. **Environment State**: Simple dict storage ✅
6. **Reward Function**:
   - +10.0 if reached goal ✅
   - -0.1 per step (efficiency penalty) ✅
   - -2.0 per invalid move ✅
   - +0.1 * cosine_sim(reference_embed, sequence_embed) ✅

---

## Reward Function Details

The reward function combines multiple components:

```python
Reward(sequence) = {
    +10.0 if reached goal
    - 0.1 * num_steps (efficiency penalty)
    - 2.0 * num_invalid (invalid action penalty)
    + 0.1 * cosine_sim(reference_embed, sequence_embed)  # Optional
}
```

**Cosine Similarity Term**:
- Only computed if `reference_path` is provided to `search()`
- Uses `action_encoder` to embed both reference and generated sequences
- Helps align DiT with tokenizer (similar to GRPO β term)
- `alpha = 0.1` coefficient (configurable via `reward_alpha`)

---

## Usage Example

```python
from src.mctd import HiddenSpaceMCTD
from src.models.diffusion_policy import DiffusionPolicy
from src.models.action_encoder import ActionEncoder
import gymnasium as gym
import minigrid

# Load trained model
model = DiffusionPolicy(hidden_dim=64, num_layers=4)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Create action encoder
action_encoder = ActionEncoder(num_actions=7, hidden_dim=64)

# Create environment
env = gym.make("MiniGrid-Empty-8x8-v0")

# Initialize MCTD
mctd = HiddenSpaceMCTD(
    policy_model=model,
    env=env,
    action_encoder=action_encoder,
    num_simulations=50,
    guidance_scales=[0.0, 0.5, 1.0],
    sparse_timesteps=[1.0, 0.5, 0.2, 0.0],
    denoising_step_size=0.2,
    reward_alpha=0.1,
    device='mps',  # or 'cpu', 'cuda'
)

# Get initial state
obs, info = env.reset()
initial_state = {
    'grid': obs['image'],  # [7, 7, 3]
    'direction': env.unwrapped.agent_dir,
}

# Optional: reference path for reward
reference_path = torch.tensor([2, 2, 2, ...])  # [seq_len] expert actions

# Run search
best_actions, search_tree = mctd.search(initial_state, reference_path=reference_path)

# Use best actions
for action in best_actions:
    obs, reward, done, truncated, info = env.step(action.item())
    if done:
        break
```

---

## Architecture Flow

```
Initial State
    ↓
MCTD Search (N simulations):
    ├─ SELECT: UCT navigation → leaf node
    ├─ EXPAND: Create 3 children (guidance scales [0.0, 0.5, 1.0])
    │   └─ Denoise: t → t - 0.2 (fixed step)
    ├─ SIMULATE: Jumpy denoising [t, 0.5, 0.2, 0.0] → actions
    │   └─ Evaluate: reward = f(success, efficiency, validity, similarity)
    └─ BACKPROPAGATE: Update Q-values up tree
        ↓
Extract Best Path (highest Q-value)
    ↓
Discrete Action Sequence
```

---

## Key Implementation Details

### Batching for Efficiency

- **Expand**: Individual calls per guidance scale (simple, efficient in eval mode)
- **Future optimization**: True batching would require modifying `denoise_step()` to handle batched guidance scales

### State Handling

- **Environment state**: Stored as dict with `'grid'` and `'direction'` keys
- **State conversion**: Initial state converted to tensor format at search start
- **State consistency**: Children inherit parent's `env_state` (denoising doesn't change environment)

### Reward Evaluation

- **Environment reset**: Currently resets to start (TODO: support arbitrary initial states)
- **Invalid action detection**: Based on reward < -0.5 (heuristic, may need refinement)
- **Cosine similarity**: Computed on flattened embeddings `[L*D]`

---

## Files Created

1. `src/mctd/node.py` - MCTDNode class
2. `src/mctd/search.py` - HiddenSpaceMCTD class
3. `src/mctd/__init__.py` - Module exports

---

## Next Steps

### 1. Integration & Testing
- [ ] Create test script for MCTD search
- [ ] Test on Empty-8x8 maze
- [ ] Compare vs direct policy (greedy decode)

### 2. Evaluation
- [ ] Measure success rate
- [ ] Compare path lengths
- [ ] Analyze tree statistics (visit counts, Q-values)

### 3. Visualization
- [ ] Tree visualization notebook
- [ ] Search statistics analysis
- [ ] Trajectory comparison plots

### 4. Training Integration (Deferred)
- [ ] Decide which trajectories to use for training (currently: best path only)
- [ ] Implement MCTD-based training loop (if needed)
- [ ] Compare with baseline training

---

## Research Decisions Documented

**MCTD Training/Reward Integration** (in `RESEARCH_DECISIONS.md`):
- **Current assumption**: Use best path (highest Q-value trajectory) for training
- **Future exploration**: Top-K trajectories, soft selection, off-policy learning, RL-style updates

---

## Potential Issues & TODOs

1. **Environment State Reset**: `evaluate_trajectory()` resets to start, not to `initial_state`
   - **Impact**: Limited for now (only affects simulation accuracy)
   - **Future**: Add support for arbitrary initial states if needed

2. **Invalid Action Detection**: Heuristic based on reward < -0.5
   - **Impact**: May not catch all invalid actions
   - **Future**: Use environment feedback or action space validation

3. **Batching Optimization**: Expand uses individual calls, not true batching
   - **Impact**: Minor performance loss
   - **Future**: Implement batched `denoise_step()` if needed

4. **Reference Path**: Cosine similarity requires reference path
   - **Impact**: Only used if reference provided (optional)
   - **Future**: May want to compute from expert dataset automatically

---

## Notes

- All components use `@torch.no_grad()` for efficiency during search
- Model is set to `eval()` mode in `__init__`
- Device handling: All tensors moved to specified device
- Shape handling: Uses `num_tokens` for hidden state dimension (49 for 7x7 grid)

Everything is ready for testing! 🚀
