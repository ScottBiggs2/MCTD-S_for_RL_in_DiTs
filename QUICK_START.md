# Quick Start Guide

## Understanding the Project

This project implements **Monte Carlo Tree Diffusion (MCTD)** for maze navigation as a testbed for diffusion language models. The key innovation is searching over continuous hidden representations that denoise into discrete action sequences.

**Research Goal**: Validate MCTD techniques on mazes before scaling to full language reasoning tasks.

## Current Status

✅ **Initial Setup Complete**
- Project structure created
- Environment wrapper implemented
- Shape verification utilities ready
- Test scripts prepared

⏸️ **Waiting for Research Decisions**
- See `RESEARCH_DECISIONS.md` for choices needed
- Critical: Model architecture (hidden_dim, num_layers)

## Next Steps

### 1. Review Research Decisions
Open `RESEARCH_DECISIONS.md` and confirm:
- Model architecture size
- Expert data generation method
- Number of trajectories
- Max sequence length

### 2. Test Current Setup
```bash
# Install dependencies first
pip install -r requirements.txt

# Test environment
python scripts/test_environment_setup.py
```

### 3. After Decisions Confirmed
1. Implement encoders (ActionEncoder, StateEncoder)
2. Generate expert data
3. Verify data shapes for DiT training

## Key Files

- `CHECKLIST.md`: Complete task breakdown
- `RESEARCH_DECISIONS.md`: Decisions needing your input
- `SETUP_SUMMARY.md`: What's been completed
- `maze_mctd_plan.txt`: Full implementation plan

## Shape Verification

All tensor shapes must be compatible with DiT training:
- **States**: `[batch, 147]` (flattened 7x7x3 grid)
- **Actions**: `[batch, seq_len]` (discrete action indices)
- **Hidden Actions**: `[batch, seq_len, hidden_dim]` (for DiT input)

Use `src/utils/shape_verification.py` to verify shapes at each step.

## Architecture Overview

```
Maze Observation → State Encoder → State Embedding
                                              ↓
Action Sequence → Action Encoder → Hidden Actions → DiT → Denoised Actions
                                              ↑
                                    (MCTD searches here)
```

The key insight: MCTD searches in the continuous hidden space, not discrete actions.
