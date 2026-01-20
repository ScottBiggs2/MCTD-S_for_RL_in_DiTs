# Research Decisions - To Be Confirmed

This document tracks research choices that need your input before proceeding.

## Phase 1: Foundation Decisions

### 1. Model Architecture (Phase 1-2)
**Question**: What model size should we use for Phase 1-2 local development?

**Plan Suggestion**:
- Hidden dim: 64-128
- Layers: 2-4
- Parameters: ~100K-500K

**Options**:
- [ ] Use plan suggestion (64-128 hidden_dim, 2-4 layers)
- [ ] Smaller (32-64 hidden_dim, 2 layers) - faster iteration
- [X] Larger (128-256 hidden_dim, 4-6 layers) - better quality but slower

**Recommendation**: Start with plan suggestion (128 hidden_dim, 4 layers) for good balance.
I agree with the recommendation. If we see performance is poor or that it is too slow, we can easily change it later. 
---

### 2. Expert Data Generation Method
**Question**: Which solver should we use for expert trajectories?

**Options**:
- [X] BFS (Breadth-First Search) - guaranteed optimal, simpler
- [ ] A* (A-star) - optimal with heuristics, more efficient
- [ ] Hand-coded policy - faster but may not be optimal

**Plan Suggestion**: BFS for simplicity and guaranteed optimality.

**Recommendation**: Start with BFS, can switch to A* if performance is an issue.
I agree with the recommendation. Starting with BFS is easy, it's lightweight, and we don't have any heuristics to consider at this phase. Hand-coding a policy seems silly for this task. 
---

### 3. Data Collection Parameters
**Question**: How many expert trajectories per environment?

**Plan Suggestion**:
- Empty-8x8: 500+ episodes
- FourRooms: 500+ episodes

**Options**:
- [X] Use plan suggestion (500 per env)
- [ ] More (1000+) - better coverage but slower
- [ ] Fewer (100-250) - faster but may limit training quality

**Recommendation**: Start with 500, can generate more if needed.
I agree with the recommendation. We can always go back later and collect more data later, but starting small keeps our velocity up. We will need to be sure to curate an isolated test set of environments to evaluate model performance in. 
---

### 4. State Representation
**Question**: How should we represent MiniGrid observations?

**Plan Suggestion**:
- Grid: flattened 7x7x3 observation
- Direction: separate embedding (0-3)
- Position: if available

**Options**:
- [ ] Use plan suggestion (flattened grid + direction)
- [X] Use CNN encoder for grid (better spatial understanding)
- [ ] Include additional features (agent position, inventory, etc.)

**Recommendation**: Start with plan suggestion, can enhance later.
Since our DiT models are small, CNN encoding/decoding could help boost their performance. I think we should use lightweight ResNet style encoder/decoders to tokenize states. We should then add position embeddings between the tokens and the DiT core. ie: State -> CNN Tokenizer -> rep(s)+pos. embed. -> DiT -> rep(s) -> CNN Tokenizer -> Action. We could fairly easily revisit this question later, but this seems reasonable to me at this phase. 
---

### 5. Action Sequence Length
**Question**: What maximum sequence length for action sequences?

**Plan Suggestion**: max_seq_len=64

**Options**:
- [ ] 32 - shorter, faster training
- [X] 64 - plan suggestion
- [ ] 128 - longer trajectories, more memory

**Recommendation**: Start with 64, can adjust based on actual trajectory lengths.
I agree with the recommendation. If we find that the trajectories are typically shorter, it won't be too difficult to go back later and reduce the max_seq_len. 
---

### 6. Masking Schedule (Phase 2)
**Question**: Which masking schedule for diffusion training?

**Options**:
- [ ] Linear: mask_ratio = t
- [X] Cosine: mask_ratio = cos(π/2 * (1-t))
- [ ] Constant: mask_ratio = 0.15 (BERT-style)

**Plan Suggestion**: Cosine schedule

**Recommendation**: Start with cosine, can experiment later.
I agree with the recommendation. Cosine scheduling seems like the correct scheduler choice for this scale. 
---

## Decisions Needed Before Phase 1 Implementation

**Critical (needed now)**:
1. Model architecture size (hidden_dim, num_layers)
2. Expert data generation method (BFS vs A*)
3. Number of trajectories per environment
4. Max sequence length

**Can defer to Phase 2**:
- Masking schedule
- Learning rate, batch size
- Other training hyperparameters

---

## Current Status
- [X] Decisions confirmed
- [X] Ready to proceed with Phase 1

---

## Phase 3: MCTD Search Decisions

### MCTD Training/Reward Integration
**Question**: How does MCTD search integrate with model training? Which trajectory from the search tree should we use for training?

**Current Assumption**: Use the expected best path (highest Q-value trajectory) from the search tree.

**Context**: 
- MCTD search builds a tree of possible action sequences
- Each trajectory has a Q-value (expected reward)
- For training, we need to select which trajectory(s) to use

**Options**:
- [X] Best path only (highest Q-value trajectory) - **Current assumption**
- [ ] Top-K trajectories (weighted by Q-value or visitation count)
- [ ] All trajectories (weighted by visitation probability)
- [ ] Soft selection (sample according to visit distribution)
- [ ] Off-policy learning (use all rollouts regardless of best path)

**Considerations**:
- **Best path only**: Simple, focuses on exploiting good solutions
- **Top-K**: More diverse training data, still focused on high-quality paths
- **All trajectories**: Maximum data usage, includes exploration
- **Soft selection**: Matches tree search behavior more closely
- **Off-policy**: Learn from all search experience, not just best decisions

**Future Exploration**:
- RL-style training (use tree statistics as targets)
- GRPO-style updates (reward-weighted policy optimization)
- Adversarial training (use tree visit patterns to guide masking)
- Imitation learning from search (learn to predict search decisions)

**Recommendation**: Start with best path only (simplest baseline), experiment with alternatives once core MCTD is working.

---

## Additional Research Questions (To Defer)

### BFS Optimization
**Question**: The current BFS implementation creates a new environment for each action exploration, which is slow but correct. Should we optimize?

**Options**:
- [X] Keep current implementation (works but slow, ~500 episodes may take time)
- [ ] Optimize with proper state cloning mechanisms
- [ ] Use environment seeds + state hashing for faster exploration

**Current**: Using simple approach that creates new environments. Works correctly but may be slow.

**Recommendation**: Start with current implementation, optimize if data generation is too slow.
After running `python scripts/generate_expert_data.py` to generate 500 traces, I think the current approach is fine. 500 trajectories only took 3-4 minutes to collect, and the trajectories are independent. A linear scaling of ~2 trajectory per second is acceptable for the forseeable future. 
---

### Token Aggregation for Actions
**Question**: ActionDecoder outputs (batch, 49, 7) - one action per token. For action sequences of length 64, we currently take the first 64 tokens. Should we use a different aggregation?

**Options**:
- [X] Keep current approach (first N tokens)
- [ ] Use attention/pooling to aggregate all tokens
- [ ] Use learned aggregation mechanism

**Current**: Using first seq_len tokens from decoder output.

**Recommendation**: Start with current approach, can refine based on training results.
I doubt the traces will be longer than 64 tokens, so the current approach is fine. In fact, while generating the expert data, the empty 8x8 average trajectory length is only 12. 

In the four rooms environment, the BFS solver seems to fail about 15% of the time. I'm not sure how serious of an issue this is, but I'd like you to investigate the current implementation and any discussion online or in the documentation. The average time per trace in the four rooms environment seems to be about 1-1.5 traces per second, which is acceptable at the present scale.  