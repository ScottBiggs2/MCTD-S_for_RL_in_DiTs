# Maze-MCTD Implementation Checklist

## Project Overview
Monte Carlo Tree Diffusion (MCTD) in continuous hidden space for discrete maze navigation. This serves as a testbed for diffusion language model (dLLM) techniques before scaling to full language tasks.

**Core Innovation**: Search over continuous hidden representations that denoise into discrete action sequences.

---

## Phase 1: Foundation (Week 1) - M1 MacBook
**Goal**: Get basic environment working and collect expert data.

### 1.1 Environment Setup
- [x] Install dependencies (torch, gymnasium, minigrid, etc.) - requirements.txt created
- [ ] Verify M1 MacBook MPS backend availability - **TODO: Run test script**
- [ ] Test MiniGrid environment import and basic functionality - **TODO: Run test script**
- [x] Create project directory structure - **COMPLETED**

### 1.2 MiniGrid Wrapper Implementation
- [x] Implement `MazeEnvironment` class in `src/environments/minigrid_wrapper.py`
  - [x] State extraction from MiniGrid observations
  - [x] Action space handling (7 discrete actions)
  - [x] State embedding conversion
  - [x] Branch point detection logic (basic heuristic)
- [x] Write unit tests for wrapper - test_environment_setup.py created
- [x] Verify state shape consistency - shape_verification.py utilities created

### 1.3 Expert Data Generation
- [x] Implement BFS solver in `scripts/generate_expert_data.py`
- [ ] Generate expert trajectories for:
  - [ ] MiniGrid-Empty-8x8-v0 (500 episodes) - **TODO: Run script**
  - [ ] MiniGrid-FourRooms-v0 (500 episodes) - **TODO: Run script**
- [x] Save trajectories with proper format:
  - [x] States (grid, direction)
  - [x] Actions (discrete indices)
  - [x] Trajectory metadata (length, success)
  - [x] Train/test split (80/20)
- [ ] Verify data format and shapes - **TODO: After data generation**

### 1.4 Action & State Encoders
- [x] Implement `ActionEncoder` in `src/models/action_encoder.py`
  - [x] Discrete action → continuous embedding
  - [x] Decode hidden states → action logits
  - [x] Verify embedding dimensions
- [x] Implement `StateEncoder` in `src/models/state_encoder.py`
  - [x] CNN tokenizer for grid (ResNet-style)
  - [x] Direction embedding
  - [x] Position embeddings
  - [x] Combined state representation
- [x] Implement `ActionDecoder` with CNN tokenizer
- [x] Write tests for encoders - test_cnn_architecture.py created
- [x] **Verify tensor shapes match DiT requirements** - CNN token shapes verified

### 1.5 Data Validation & Shape Verification
- [x] Create data loading utilities - shape_verification.py created
- [x] Verify batch shapes:
  - [x] States: [batch, hidden_dim] or [batch, seq_len, hidden_dim] - utilities ready
  - [x] Actions: [batch, seq_len] - utilities ready
  - [x] Hidden action embeddings: [batch, seq_len, hidden_dim] - utilities ready
- [ ] Test data pipeline end-to-end - **TODO: After data collection**
- [ ] Create visualization notebook for trajectories - **TODO: After data collection**

### 1.6 Visualization Tools
- [ ] Create `notebooks/01_environment_exploration.ipynb`
  - [ ] Environment rendering
  - [ ] Trajectory visualization
  - [ ] State representation inspection
  - [ ] Branch point visualization

---

## Phase 2: Baseline Diffusion Model (Week 2) - M1 MacBook
**Goal**: Train vanilla masked diffusion policy on expert trajectories.

### 2.1 DiT Architecture Components
- [ ] Implement `SinusoidalPositionEmbedding` for timesteps
- [ ] Implement `DiTBlock` with adaptive layer norm
- [ ] Implement `DiffusionPolicy` model
  - [ ] State encoder integration
  - [ ] Action encoder integration
  - [ ] Timestep conditioning
  - [ ] Position embeddings
  - [ ] Transformer blocks
  - [ ] Output head (action logits)

### 2.2 Masked Diffusion Training
- [ ] Implement `MaskedDiffusionTrainer` in `src/training/mdlm_trainer.py`
  - [ ] Masking schedule (linear/cosine/constant)
  - [ ] Noise injection for masked positions
  - [ ] Loss computation (only on masked tokens)
  - [ ] Training loop
  - [ ] Validation loop

### 2.3 Training Infrastructure
- [ ] DataLoader setup (num_workers=0 for M1)
- [ ] Optimizer configuration (AdamW)
- [ ] Learning rate scheduling
- [ ] Checkpointing utilities
- [ ] Wandb/Tensorboard logging

### 2.4 Evaluation
- [ ] Evaluation script for test mazes
- [ ] Metrics: success rate, path length, accuracy
- [ ] Baseline comparison

---

## Phase 3: MCTD Search in Hidden Space (Week 3) - M1 MacBook
**Goal**: Implement tree search over hidden action sequences.

### 3.1 MCTS Tree Structure
- [ ] Implement `MCTDNode` in `src/mctd/node.py`
  - [ ] Hidden state storage
  - [ ] Noise level tracking
  - [ ] Environment state storage
  - [ ] Tree statistics (visits, Q-value)
  - [ ] UCT scoring

### 3.2 MCTD Search Algorithm
- [ ] Implement `HiddenSpaceMCTD` in `src/mctd/search.py`
  - [ ] Selection phase (UCT)
  - [ ] Expansion phase (different guidance scales)
  - [ ] Simulation phase (jumpy denoising)
  - [ ] Backpropagation phase
  - [ ] Best trajectory extraction

### 3.3 Integration & Testing
- [ ] Connect MCTD to trained diffusion policy
- [ ] Test search on Empty-8x8
- [ ] Compare vs direct policy
- [ ] Tree visualization

---

## Phase 4: Hierarchical Planning (Week 4) - Cloud GPU
**Goal**: Implement planner/executor separation for speedup.

### 4.1 Hierarchical Models
- [ ] Implement `WaypointPlanner` (coarse waypoints)
- [ ] Implement `ActionExecutor` (fine-grained actions)
- [ ] Train both models

### 4.2 Hierarchical MCTD
- [ ] Integrate planner into MCTD
- [ ] Waypoint-level search
- [ ] Action-level execution
- [ ] Speedup measurements

---

## Phase 5: Adversarial Masking (Week 5) - Cloud GPU
**Goal**: Use tree statistics to guide adversarial training.

### 5.1 Tree Statistics Collection
- [ ] Visit count tracking
- [ ] Importance map computation
- [ ] Branch point identification

### 5.2 Adversarial Training
- [ ] Implement `AdversarialMCTDTrainer`
- [ ] Tree-guided mask generation
- [ ] Training loop integration
- [ ] Robustness evaluation

---

## Phase 6: Parallel Optimization (Week 6) - Cloud GPU
**Goal**: Achieve 10-50x speedup via parallelization.

### 6.1 Parallel MCTD
- [ ] Batch parallel rollouts
- [ ] Delayed tree updates
- [ ] Redundancy-aware selection (RAS)

### 6.2 Benchmarks
- [ ] Speedup measurements
- [ ] Quality vs speed tradeoffs
- [ ] Comprehensive evaluation

---

## Current Focus: Phase 1 Initial Setup

### Immediate Tasks
1. **Environment Setup**
   - Install dependencies
   - Verify M1 MPS backend
   - Create directory structure

2. **Data Collection & Validation**
   - Implement expert trajectory generation
   - Verify data shapes
   - Test data loading pipeline

3. **Shape Verification for DiT**
   - Ensure all tensor shapes are compatible
   - Document expected shapes
   - Create shape validation utilities

---

## Research Decisions to Defer
- Model architecture choices (hidden_dim, num_layers, num_heads)
- Masking schedule (linear vs cosine vs constant)
- Guidance scales for MCTD expansion
- Number of simulations for tree search
- Training hyperparameters (learning rate, batch size)

---

## Notes
- All Phase 1-3 work on M1 MacBook (small models, fast iteration)
- Phase 4-6 transition to Cloud GPU (larger models)
- Use num_workers=0 for DataLoader on M1
- Keep models small during local development (~100K-2M params)
