# Current Status & Next Steps

## Phase 1: Foundation - Nearly Complete ✅

### ✅ Completed
1. **Environment Setup**
   - ✅ Project structure created
   - ✅ Dependencies listed in requirements.txt
   - ✅ MiniGrid wrapper implemented and tested

2. **CNN Architecture**
   - ✅ StateEncoder with CNN tokenizer (ResNet-style)
   - ✅ ActionEncoder/Decoder with CNN tokenizer
   - ✅ Position embeddings
   - ✅ Shape verification utilities
   - ✅ All shapes verified for DiT compatibility

3. **Expert Data Generation**
   - ✅ BFS solver implemented (fixed to use full state)
   - ✅ Train/test split (80/20)
   - ✅ Trajectory dataset class

### ⏳ Remaining Phase 1 Tasks

1. **Generate Expert Data** (Immediate)
   ```bash
   python scripts/generate_expert_data.py
   ```
   - Should now work correctly with fixed BFS
   - Will create: `data/Empty-8x8_train.pkl`, `data/Empty-8x8_test.pkl`, etc.
   - Expected: 500/500 for Empty-8x8, ~495-500/500 for FourRooms

2. **Verify Data Pipeline** (After data generation)
   - Load dataset and verify shapes
   - Check trajectory length statistics
   - Validate train/test split
   - Create data loading script/test

3. **Visualization Notebook** (Optional but useful)
   - `notebooks/01_environment_exploration.ipynb`
   - Environment rendering
   - Trajectory visualization
   - State representation inspection

---

## Phase 2: Baseline Diffusion Model - Next Major Phase

**Goal**: Train vanilla masked diffusion policy on expert trajectories.

### 2.1 DiT Architecture Components (First priority)

Need to implement:

1. **`SinusoidalPositionEmbedding`** for timesteps
   - Location: `src/models/components/timestep_embed.py`
   - Maps diffusion timestep `t ∈ [0, 1]` → embedding

2. **`DiTBlock`** with adaptive layer norm
   - Location: `src/models/components/dit_block.py`
   - Transformer block with:
     - Self-attention
     - MLP
     - Adaptive LayerNorm (conditioned on timestep embedding)

3. **`DiffusionPolicy`** model
   - Location: `src/models/diffusion_policy.py`
   - Integrates:
     - StateEncoder (already done ✅)
     - ActionEncoder (already done ✅)
     - Timestep embeddings
     - Position embeddings
     - DiT blocks (4 layers, 128 hidden_dim)
     - Output head → action logits

**Architecture Flow**:
```
Input State (grid + direction)
    ↓
StateEncoder (CNN + Pos Embed)  ✅
    ↓
Tokens: (batch, 49, 128)
    ↓
+ Timestep Embedding (t)
    ↓
DiT Blocks (4 layers)  ← TO IMPLEMENT
    ↓
Tokens: (batch, 49, 128)
    ↓
ActionDecoder (CNN)  ✅
    ↓
Action Logits: (batch, 64, 7)
```

### 2.2 Masked Diffusion Training

Need to implement:

1. **`MaskedDiffusionTrainer`**
   - Location: `src/training/mdlm_trainer.py`
   - Features:
     - Cosine masking schedule (confirmed)
     - Noise injection for masked positions
     - Loss on masked tokens only
     - Training/validation loops

### 2.3 Training Infrastructure

1. **DataLoader setup**
   - Use `TrajectoryDataset` (already done ✅)
   - `num_workers=0` for M1 MacBook
   - Batch size: ~16-32

2. **Optimizer & Training**
   - AdamW optimizer
   - Learning rate: TBD (need research decision)
   - Checkpointing utilities
   - Wandb logging

### 2.4 Evaluation

1. **Evaluation script**
   - Test on validation set
   - Metrics: success rate, path length, action accuracy
   - Compare vs expert trajectories

---

## Architecture Decisions (Confirmed)

From `RESEARCH_DECISIONS.md`:

- **Hidden Dimension**: 128 ✅
- **Number of Layers**: 4 ✅
- **Number of Tokens**: 49 (7x7 grid) ✅
- **Max Sequence Length**: 64 ✅
- **Masking Schedule**: Cosine ✅
- **State Representation**: CNN tokenizer (ResNet-style) ✅

## Still Need Research Decisions (Phase 2)

1. **Training Hyperparameters**:
   - Learning rate: ? (typically 1e-4 to 1e-3)
   - Batch size: ? (16-32 for M1)
   - Weight decay: ? (typically 0.01-0.1)
   - Number of epochs: ? (until convergence)

2. **Diffusion Schedule**:
   - Number of diffusion steps: ? (typically 50-1000)
   - Noise schedule: ? (linear, cosine, etc.)

3. **Model Output**:
   - How to handle 49 tokens → 64 action sequence?
   - Current: Take first 64 tokens (confirmed for now)

---

## Recommended Next Steps

### Immediate (Finish Phase 1)
1. **Generate expert data** with fixed BFS
   ```bash
   python scripts/generate_expert_data.py
   ```

2. **Create data verification script**
   - `scripts/verify_data.py`
   - Load and check dataset
   - Print statistics (lengths, success rates)
   - Verify shapes match architecture

3. **Optional: Visualization notebook**
   - Quick exploration of collected data

### Then (Start Phase 2)
1. **Implement DiT components** (in order):
   - `SinusoidalPositionEmbedding`
   - `DiTBlock`
   - `DiffusionPolicy` (integrates everything)

2. **Implement training**:
   - `MaskedDiffusionTrainer`
   - Training infrastructure
   - Evaluation metrics

3. **Train baseline model**:
   - Start with Empty-8x8
   - Evaluate on test set
   - Iterate on hyperparameters

---

## Files to Create Next

### Phase 1 Completion
- [ ] `scripts/verify_data.py` - Data verification script

### Phase 2 Implementation
- [ ] `src/models/components/timestep_embed.py` - Timestep embeddings
- [ ] `src/models/components/dit_block.py` - DiT transformer block
- [ ] `src/models/diffusion_policy.py` - Full diffusion policy
- [ ] `src/training/mdlm_trainer.py` - Masked diffusion trainer
- [ ] `src/utils/checkpointing.py` - Model checkpointing
- [ ] `src/utils/logging.py` - Wandb/tensorboard logging
- [ ] `scripts/train_baseline.py` - Training script
- [ ] `scripts/evaluate.py` - Evaluation script

---

## Timeline Estimate

- **Phase 1 Completion**: 1-2 hours (data generation + verification)
- **Phase 2 Implementation**: 1-2 days (DiT + training code)
- **Phase 2 Training**: 1-2 days (training + debugging)

**Total to working baseline**: ~1 week

---

## Key Insights from Phase 1

1. **CNN Tokenizers**: Working well, shapes verified
2. **BFS Fix**: Critical - partial observations don't work for planning
3. **State Representation**: Position + direction for BFS, CNN tokens for DiT
4. **Data Format**: States as dicts with 'grid' and 'direction', actions as arrays

Everything is set up correctly for Phase 2!
