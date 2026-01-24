# Diffusion Language Model Research Plan
## Converting Qwen3-0.6B Base to a Diffusion Language Model

**Author**: Scott  
**Date**: January 2026  
**Base Model**: [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)

---

## Executive Summary

This document outlines a research plan for adapting Qwen3-0.6B-Base into a diffusion language model (dLLM) using established AR→Diffusion conversion recipes. The primary approach leverages **masked diffusion fine-tuning** rather than full pretraining, following successful patterns from dLLM's Tiny-A2D, Salesforce CoDA, and Apple DiffuCoder.

**Key insight from existing work**: Converting AR models to diffusion requires only ~1B tokens of fine-tuning (a 500× reduction vs. from-scratch diffusion pretraining), while preserving most of the original model's capabilities.

---

## Table of Contents

1. [Background & Theory](#1-background--theory)
2. [Repository & Paper Analysis](#2-repository--paper-analysis)
3. [Development Plan](#3-development-plan)
4. [DrOPE Integration](#4-drope-integration)
5. [Training Infrastructure](#5-training-infrastructure)
6. [Evaluation Strategy](#6-evaluation-strategy)
7. [Research Uncertainties](#7-research-uncertainties)
8. [Future Directions](#8-future-directions)

---

## 1. Background & Theory

### 1.1 Discrete Diffusion vs. Flow Matching

Most "diffusion" language models are technically closer to **discrete flow matching**—they learn vector fields pointing toward the data manifold rather than performing true iterative denoising. The key mathematical frameworks:

| Approach | Description | Key Papers |
|----------|-------------|------------|
| **MDLM** (Masked Diffusion) | Random token masking, predict masked tokens | SMDM, LLaDA |
| **BD3LM** (Block Diffusion) | Block-wise masked diffusion with semi-autoregressive decoding | BD3LM, Fast-dLLM v2 |
| **Edit Flows** | Discrete flow matching with insertion/deletion/substitution | Edit Flows |

### 1.2 Why AR→Diffusion Conversion Works

Autoregressive models already possess:
- Strong token prediction capabilities (the core diffusion objective)
- Bidirectional context understanding in intermediate layers
- Vocabulary and tokenization that transfers directly

**The adaptation process** primarily teaches the model to:
1. Handle bidirectional attention (vs. causal masking)
2. Predict tokens from any masked position (vs. left-to-right only)
3. Generate in parallel across masked positions

### 1.3 Qwen3-0.6B Architecture Quick Reference

```
Model: Qwen3-0.6B-Base
Parameters: ~600M
Layers: 28
Hidden Dim: 1024
Attention Heads: 16
Context Length: 32,768
Vocabulary: 151,936 tokens
Position Embedding: RoPE
```

---

## 2. Repository & Paper Analysis

### 2.1 ZHZisZZ/dLLM (Primary Reference)

**Repository**: https://github.com/ZHZisZZ/dllm

**Why it's important**: Provides the most accessible and well-documented AR→Diffusion conversion recipe, with working Qwen3-0.6B checkpoints.

**Key Components**:
```
dllm/
├── core/
│   ├── trainers/
│   │   ├── MDLMTrainer      # Masked Diffusion trainer
│   │   └── BD3LMTrainer     # Block Diffusion trainer
│   ├── samplers/
│   │   ├── MDLMSampler      # Inference for masked diffusion
│   │   └── BD3LMSampler     # Inference for block diffusion
│   └── schedulers/          # Noise/masking schedules
├── pipelines/
│   └── a2d/                 # AR-to-Diffusion conversion
└── examples/
    └── a2d/                 # Training scripts for conversion
```

**Training Recipe (from Tiny-A2D)**:
- Dataset: Mixed pretraining corpus (~10-20B tokens)
- Training: MDLM or BD3LM objective
- Duration: ~120K steps on 8 GPUs
- Result: `dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1`

**Key Findings from Tiny-A2D**:
1. Block diffusion (BD3LM) generally outperforms vanilla masked diffusion (MDLM) on math/code
2. The quality of the base AR model matters more than adaptation compute
3. Small models (0.5-0.6B) can achieve competitive diffusion performance

### 2.2 Salesforce CoDA

**Repository**: https://github.com/SalesforceAIResearch/CoDA  
**Paper**: https://arxiv.org/abs/2510.03270

**Why it's important**: Demonstrates efficient TPU training pipeline and sophisticated masking strategies for code-focused diffusion models.

**Three-Stage Training Pipeline**:

| Stage | Tokens | Data | Purpose |
|-------|--------|------|---------|
| **Pre-training** | 179B | Web + code (dclm, Stack v2, RedPajama) | Foundation |
| **Mid-training** | 20B | ArXiv, Gutenberg, OpenCoder, SmolLM-PythonEdu | Progressive masking curriculum |
| **Post-training** | ~1B | OpenCoder Stage 1 & 2 | Instruction tuning |

**Masking Strategies (Critical Innovation)**:
```python
# S1: Unmaskable Prefix - Never mask the prompt/instruction
# Ensures stable conditioning on initial context

# S2: Truncated Suffix - Randomly truncate sequence length
# Teaches handling of variable-length sequences

# S3: Block Masking - Mask contiguous spans
# Simulates realistic infilling/code-repair scenarios
```

**Curriculum Schedule**:
- Gradually increase masking strategy probabilities over epochs
- Transitions from random token masking → structured code infilling
- Aligns internal noise distribution with inference behavior

### 2.3 Apple DiffuCoder

**Repository**: https://github.com/apple/ml-diffucoder  
**Paper**: https://arxiv.org/abs/2506.20639

**Why it's important**: Introduces autoregressiveness analysis and Coupled-GRPO for post-training optimization.

**Key Innovations**:

**1. Autoregressiveness Score**
- Quantifies the causal (left-to-right) pattern during generation
- Code tasks induce less global AR-ness than math tasks
- Temperature affects not just sampled tokens but generation ORDER

**2. Coupled-GRPO (Post-training)**
```python
# Standard per-timestep loss only computes log-probs at masked positions
# → High variance, inefficient learning

# Coupled-GRPO solution:
# For each example, select λ pairs of timesteps (t, t̂) where t + t̂ = T
# Apply complementary masks that together cover ALL tokens
# Every token gets computed exactly once across the pair
```

**Inference Configuration**:
```python
output = model.diffusion_generate(
    input_ids,
    max_new_tokens=256,
    steps=256,              # diffusion timesteps
    temperature=0.2,        # sampling temperature
    top_p=0.95,            # nucleus sampling
    alg="entropy",         # entropy-based token selection
    alg_temp=0.,           # algorithm temperature
)
```

### 2.4 Fast-dLLM v2 (Efficiency Reference)

**Model**: https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B  
**Paper**: https://arxiv.org/abs/2509.26328

**Key Efficiency Insights**:
- Only ~1B tokens needed for AR→Diffusion fine-tuning
- Complementary attention mask enables blockwise bidirectional context
- Hierarchical caching: block-level + sub-block level
- Up to 2.54× throughput improvement

---

## 3. Development Plan

### Phase 0: Environment Setup (Days 1-2)

**Objective**: Establish reproducible development environment

```bash
# Clone and setup dLLM
git clone https://github.com/ZHZisZZ/dllm
cd dllm

conda create -n dllm python=3.10 -y
conda activate dllm
conda install cuda=12.4 -c nvidia

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -e .

# Evaluation harness
git submodule update --init --recursive
pip install -e "lm-evaluation-harness[ifeval,math]"
```

**Validation**:
```bash
# Test existing checkpoint works
python -u examples/a2d/mdlm/chat.py \
    --model_name_or_path dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1 \
    --chat_template True
```

---

### Phase 1: Baseline Conversion (Days 3-7)

**Objective**: Replicate Tiny-A2D results with vanilla MDLM

**Step 1.1: Data Preparation**
```bash
# Preprocess dataset for MDLM training
python dllm/tools/preprocess_sft_dataset.py \
    --model_name_or_path "Qwen/Qwen3-0.6B-Base" \
    --sft_map_fn_path "dllm.utils.default_mdlm_sft_map_fn" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir "data/sft/qwen3-0.6b/tulu-3" \
    --num_proc 64
```

**Step 1.2: MDLM Training**
```bash
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/a2d/mdlm/sft.py \
    --model_name_or_path "Qwen/Qwen3-0.6B-Base" \
    --dataset_args "data/sft/qwen3-0.6b/tulu-3" \
    --load_preprocessed_data True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --output_dir "outputs/qwen3-0.6b-mdlm-baseline"
```

**Checkpoint**: Functional MDLM model that can generate text via diffusion

**Success Criteria**:
- Model generates coherent text with `MDLMSampler`
- Basic benchmark scores within 10% of reference checkpoints

---

### Phase 2: Block Diffusion (BD3LM) (Days 8-14)

**Objective**: Implement BD3LM for improved math/code performance

**Why BD3LM**: Empirical evidence shows it outperforms MDLM on structured tasks, likely due to:
- Staircase attention mask preserves some left-to-right bias
- Block-wise generation better matches code/math structure

**Step 2.1: BD3LM Training**
```bash
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/a2d/bd3lm/sft.py \
    --model_name_or_path "Qwen/Qwen3-0.6B-Base" \
    --dataset_args "data/sft/qwen3-0.6b/tulu-3" \
    --load_preprocessed_data True \
    --block_size 64 \
    --num_train_epochs 4 \
    --output_dir "outputs/qwen3-0.6b-bd3lm-baseline"
```

**Step 2.2: Inference Configuration**
```python
# BD3LM-specific sampling
sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
outputs = sampler.sample(
    inputs,
    block_size=64,          # tokens per block
    remasking="low_confidence",  # remask uncertain tokens
    steps=256,
    max_new_tokens=256
)
```

**⚠️ Research Uncertainty**: Optimal block size is task-dependent:
- Smaller blocks (32-64): Better for fine-grained code
- Larger blocks (128-256): Better for prose/reasoning

**Checkpoint**: BD3LM model with block diffusion inference

---

### Phase 3: Advanced Masking Strategies (Days 15-21)

**Objective**: Implement CoDA-style masking curriculum

**⚠️ Research Intervention Required**: This phase requires significant experimentation

**Step 3.1: Implement Masking Strategies**

```python
# File: dllm/core/masking_strategies.py

import torch
import random

def unmaskable_prefix_mask(input_ids, prefix_len, mask_token_id, mask_rate=0.15):
    """S1: Never mask tokens in prefix (instruction/prompt)"""
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    # Only consider positions after prefix for masking
    maskable_region = input_ids[:, prefix_len:]
    num_to_mask = int(maskable_region.numel() * mask_rate)
    
    # Random mask in suffix only
    mask_indices = torch.randperm(maskable_region.numel())[:num_to_mask]
    flat_mask = mask.view(-1)
    flat_mask[prefix_len * input_ids.size(0) + mask_indices] = True
    
    return mask.view_as(input_ids)

def truncated_suffix_mask(input_ids, mask_token_id, max_truncate_ratio=0.3):
    """S2: Randomly truncate sequence length"""
    batch_size, seq_len = input_ids.shape
    truncate_len = int(seq_len * random.uniform(0, max_truncate_ratio))
    
    # Mask everything after truncation point
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if truncate_len > 0:
        mask[:, -truncate_len:] = True
    
    return mask

def block_masking(input_ids, mask_token_id, num_blocks=3, block_size_range=(10, 50)):
    """S3: Mask contiguous spans (realistic infilling)"""
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    for _ in range(num_blocks):
        block_size = random.randint(*block_size_range)
        start_idx = random.randint(0, max(0, seq_len - block_size))
        mask[:, start_idx:start_idx + block_size] = True
    
    return mask

class MaskingCurriculum:
    """Progressive masking curriculum from CoDA"""
    def __init__(self, total_steps):
        self.total_steps = total_steps
        
    def get_strategy_probs(self, current_step):
        """Gradually shift from random → structured masking"""
        progress = current_step / self.total_steps
        
        # Early: mostly random masking
        # Late: mostly structured (S1, S2, S3) masking
        return {
            'random': max(0.1, 1.0 - progress),
            'unmaskable_prefix': min(0.3, progress * 0.4),
            'truncated_suffix': min(0.2, progress * 0.3),
            'block_masking': min(0.4, progress * 0.5)
        }
```

**Step 3.2: Integrate into Training Loop**
```python
# Modify MDLMTrainer to use curriculum
class CurriculumMDLMTrainer(MDLMTrainer):
    def __init__(self, *args, curriculum=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum = curriculum
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get current curriculum probabilities
        probs = self.curriculum.get_strategy_probs(self.state.global_step)
        
        # Sample masking strategy
        strategy = random.choices(
            list(probs.keys()), 
            weights=list(probs.values())
        )[0]
        
        # Apply selected masking
        # ... rest of training logic
```

**Checkpoint**: Model trained with progressive masking curriculum

---

### Phase 4: DrOPE Integration (Days 22-28)

**Objective**: Remove RoPE after convergence for position-agnostic diffusion

See [Section 4](#4-drope-integration) for detailed implementation plan.

---

### Phase 5: Post-Training Optimization (Days 29-35)

**Objective**: Implement Coupled-GRPO or alternative RL method

**⚠️ High Research Uncertainty**: Post-training for dLLMs is an active research area

**Option A: Coupled-GRPO (Apple DiffuCoder)**
```python
# Key insight: Standard GRPO only computes gradients at masked positions
# Coupled-GRPO uses complementary mask pairs to cover all tokens

def coupled_grpo_loss(model, input_ids, rewards, num_pairs=1):
    """
    For each training example:
    1. Select λ pairs of timesteps (t, t̂) where t + t̂ = T
    2. Apply complementary masks covering all tokens
    3. Every token gets gradient signal exactly once
    """
    T = model.config.diffusion_steps
    total_loss = 0
    
    for _ in range(num_pairs):
        t = random.randint(1, T - 1)
        t_hat = T - t
        
        # Complementary masks
        mask_t = generate_mask(input_ids, t / T)
        mask_t_hat = ~mask_t  # Complement
        
        # Forward passes with each mask
        loss_t = compute_masked_loss(model, input_ids, mask_t, rewards)
        loss_t_hat = compute_masked_loss(model, input_ids, mask_t_hat, rewards)
        
        total_loss += (loss_t + loss_t_hat) / 2
    
    return total_loss / num_pairs
```

**Option B: DPO Adaptation**
- Simpler to implement than GRPO
- May be less effective for diffusion-specific optimization

**Checkpoint**: Post-trained model with improved instruction following

---

## 4. DrOPE Integration

### 4.1 Background

**DroPE** (Sakana AI) demonstrates that positional embeddings (RoPE) can be **dropped after pretraining** with only a short recalibration phase. This is particularly relevant for diffusion models because:

1. **Bidirectional attention** doesn't require strict positional ordering
2. **Parallel generation** benefits from position-agnostic representations
3. **Context extension** becomes trivial without position extrapolation issues

### 4.2 Implementation Strategy

**Step 1: Train normally with RoPE** (Phases 1-3)
```python
# Standard Qwen3 config uses RoPE by default
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
# model.config.rope_theta = 10000.0 (default)
```

**Step 2: Remove RoPE**
```python
# After Phase 3 training converges, disable RoPE
def disable_rope(model):
    """Remove RoPE from all attention layers"""
    for layer in model.model.layers:
        # Set rotary_emb to identity/no-op
        layer.self_attn.rotary_emb = None
        
        # Or: set rope_theta to infinity (effectively disables rotation)
        # layer.self_attn.rope_theta = float('inf')
    
    return model
```

**Step 3: Short Recalibration**
```bash
# Continue training for ~2K steps without RoPE
accelerate launch examples/a2d/mdlm/sft.py \
    --model_name_or_path "outputs/qwen3-0.6b-phase3" \
    --disable_rope True \
    --max_steps 2000 \
    --learning_rate 1e-6 \  # Lower LR for fine-tuning
    --output_dir "outputs/qwen3-0.6b-drope"
```

### 4.3 Expected Behavior

From DroPE paper findings:
- **In-context perplexity**: Matches RoPE within 2K recalibration steps
- **Long-context**: Zero-shot extension without position extrapolation issues
- **Generation order**: More flexible, less left-to-right biased

### 4.4 Validation

```python
# Test position-agnostic generation
def test_drope_generation(model, tokenizer):
    prompt = "Complete this function:\ndef fibonacci(n):"
    
    # Generate with different context lengths
    for ctx_len in [512, 1024, 2048, 4096, 8192]:
        padded_prompt = "[PAD]" * (ctx_len - len(prompt)) + prompt
        output = model.diffusion_generate(padded_prompt, max_new_tokens=256)
        
        # Should produce consistent quality regardless of padding
        evaluate_output(output)
```

---

## 5. Training Infrastructure

### 5.1 Hardware Requirements

| Phase | Min GPUs | Recommended | Time Estimate |
|-------|----------|-------------|---------------|
| Phase 1 (MDLM) | 4× A100-40GB | 8× A100-80GB | 2-3 days |
| Phase 2 (BD3LM) | 4× A100-40GB | 8× A100-80GB | 2-3 days |
| Phase 3 (Curriculum) | 8× A100-40GB | 8× A100-80GB | 4-5 days |
| Phase 4 (DrOPE) | 2× A100-40GB | 4× A100-80GB | 1 day |
| Phase 5 (GRPO) | 8× A100-40GB | 8× A100-80GB | 3-5 days |

### 5.2 Distributed Training Configs

**DeepSpeed ZeRO-2** (Memory efficient, fast):
```yaml
# scripts/accelerate_configs/zero2.yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 2
  offload_optimizer_device: none
  offload_param_device: none
  gradient_accumulation_steps: 4
num_processes: 8
```

**FSDP** (Best for multi-node):
```yaml
# scripts/accelerate_configs/fsdp.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen2DecoderLayer
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_state_dict_type: FULL_STATE_DICT
num_processes: 8
```

### 5.3 Slurm Job Template

```bash
#!/bin/bash
#SBATCH --job-name=dllm-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%j.out

source ~/.bashrc
conda activate dllm

accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    --num_processes 8 \
    $@
```

---

## 6. Evaluation Strategy

### 6.1 Benchmark Suite

```bash
# Full evaluation script
bash examples/a2d/mdlm/eval.sh \
    --model_name_or_path "outputs/qwen3-0.6b-mdlm" \
    --eval_tasks "mmlu_pro,humaneval,mbpp,gsm8k,math,ifeval"
```

### 6.2 Key Metrics

| Task | Metric | Baseline (Qwen3-0.6B AR) | Target |
|------|--------|--------------------------|--------|
| MMLU-Pro | Accuracy | ~45% | >40% |
| HumanEval | pass@1 | ~35% | >30% |
| MBPP | pass@1 | ~40% | >35% |
| GSM8K | Accuracy | ~50% | >45% |
| IFEval | Strict | ~35% | >30% |

### 6.3 Diffusion-Specific Metrics

```python
def evaluate_diffusion_properties(model, test_prompts):
    """Evaluate diffusion-specific generation properties"""
    
    metrics = {}
    
    # 1. Generation diversity (should be higher than AR)
    for prompt in test_prompts:
        outputs = [model.diffusion_generate(prompt) for _ in range(10)]
        metrics['diversity'] = compute_diversity(outputs)
    
    # 2. Autoregressiveness score (from DiffuCoder)
    metrics['ar_score'] = compute_ar_score(model, test_prompts)
    
    # 3. Infilling quality (diffusion advantage)
    metrics['infill_score'] = evaluate_infilling(model, test_prompts)
    
    # 4. Parallel speedup
    metrics['speedup'] = measure_parallel_speedup(model, test_prompts)
    
    return metrics
```

---

## 7. Research Uncertainties

### 7.1 High Uncertainty (Requires Experimentation)

| Area | Uncertainty | Mitigation Strategy |
|------|-------------|---------------------|
| **Optimal block size** | Task-dependent, no universal answer | Sweep {32, 64, 128, 256} per task |
| **Masking curriculum** | CoDA curriculum may not transfer | A/B test with/without curriculum |
| **DrOPE + Diffusion** | Not validated in dLLM context | Conservative recalibration, fallback to RoPE |
| **Coupled-GRPO hyperparameters** | Limited published guidance | Start with Apple defaults, tune λ |

### 7.2 Medium Uncertainty

| Area | Uncertainty | Mitigation Strategy |
|------|-------------|---------------------|
| **Learning rate schedule** | Different from AR fine-tuning | Use dLLM defaults, monitor loss carefully |
| **Warmup ratio** | May need longer warmup for diffusion | Start with 0.1, increase if unstable |
| **Inference temperature** | Affects both tokens AND generation order | Sweep {0.1, 0.2, 0.4, 0.6} |

### 7.3 Low Uncertainty (Well-Established)

- Data preprocessing pipeline (use dLLM tools)
- Evaluation framework (use lm-evaluation-harness)
- Basic MDLM/BD3LM training objectives

---

## 8. Future Directions

### 8.1 Deferred Architecture Changes

**DyErfs (Dynamic Error Functions)** — Pinned
- Replaces LayerNorm with learnable error functions
- **Invasiveness**: Medium-High (modify all norm layers)
- **Potential benefit**: Better gradient flow, improved stability
- **When to revisit**: After Phase 3 if loss plateaus

**mHC (Manifold Hyper Connections)** — Deferred
- DeepSeek's residual connection innovation
- **Invasiveness**: Very High (restructure residual stream)
- **Potential benefit**: Stability at scale
- **When to revisit**: If scaling beyond 1B parameters

### 8.2 Advanced Masking Research

**Adversarial Masking** (From your research agenda):
- Identify and preferentially mask "important reasoning tokens"
- Approaches to explore:
  1. **Attention-based**: Mask tokens with high attention weights
  2. **Gradient-based**: Mask tokens with high gradient magnitude
  3. **Structural**: Mask tokens at reasoning-critical positions (operators, variable names)

```python
def adversarial_masking(model, input_ids, mask_rate=0.15):
    """Mask tokens that are most important for reasoning"""
    
    # Forward pass to get attention patterns
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attention = outputs.attentions[-1].mean(dim=1)  # Average over heads
    
    # Identify high-attention tokens (important for reasoning)
    attention_scores = attention.sum(dim=1)  # Aggregate attention received
    
    # Preferentially mask high-attention tokens
    num_mask = int(input_ids.numel() * mask_rate)
    _, top_indices = attention_scores.flatten().topk(num_mask)
    
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask.view(-1)[top_indices] = True
    
    return mask
```

### 8.3 Hierarchical Planning-Execution

**Concept**: Separate "planning" and "execution" models operating at different frequencies

```
┌─────────────────────────────────────────────┐
│  Planner Model (Low Frequency)              │
│  - Generates high-level reasoning skeleton  │
│  - Outputs [PLAN] tokens                    │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  Executor Model (High Frequency)            │
│  - Fills in details via diffusion           │
│  - Conditioned on [PLAN] tokens             │
└─────────────────────────────────────────────┘
```

**Research questions**:
- How to train planner and executor jointly?
- What granularity for [PLAN] tokens?
- Monte Carlo rollouts for planning?

---

## Resources

### Papers

| Paper | Link | Key Contribution |
|-------|------|------------------|
| CoDA | https://arxiv.org/abs/2510.03270 | Masking curriculum, TPU pipeline |
| DiffuCoder | https://arxiv.org/abs/2506.20639 | AR-score, Coupled-GRPO |
| DroPE | https://arxiv.org/abs/2512.12167 | Post-training PE removal |
| LLaDA | https://arxiv.org/abs/2502.09992 | Large-scale masked diffusion |
| Dream | https://arxiv.org/abs/2508.15487 | 7B diffusion LLM |
| Fast-dLLM v2 | https://arxiv.org/abs/2509.26328 | Efficient block diffusion |
| MDLM | https://arxiv.org/abs/2406.07524 | Masked diffusion foundations |
| BD3LM | https://arxiv.org/abs/2503.09573 | Block diffusion |

### Repositories

| Repo | Link | Use Case |
|------|------|----------|
| dLLM | https://github.com/ZHZisZZ/dllm | Primary training framework |
| CoDA | https://github.com/SalesforceAIResearch/CoDA | Masking strategies reference |
| DiffuCoder | https://github.com/apple/ml-diffucoder | Coupled-GRPO implementation |
| DroPE | https://github.com/SakanaAI/DroPE | PE removal reference |

### Model Checkpoints

| Checkpoint | Link | Notes |
|------------|------|-------|
| Qwen3-0.6B-Base | https://huggingface.co/Qwen/Qwen3-0.6B-Base | Starting point |
| Qwen3-0.6B-diffusion-mdlm | https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1 | Reference MDLM |
| Qwen3-0.6B-diffusion-bd3lm | https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1 | Reference BD3LM |
| DiffuCoder-7B-cpGRPO | https://huggingface.co/apple/DiffuCoder-7B-cpGRPO | GRPO reference |

---

## Appendix A: Quick Reference Commands

```bash
# Setup
git clone https://github.com/ZHZisZZ/dllm && cd dllm
conda create -n dllm python=3.10 -y && conda activate dllm
pip install -e .

# Test existing checkpoint
python -u examples/a2d/mdlm/chat.py \
    --model_name_or_path dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1

# Train MDLM from Qwen3-0.6B
accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/a2d/mdlm/sft.py \
    --model_name_or_path "Qwen/Qwen3-0.6B-Base" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --num_train_epochs 4

# Train BD3LM from Qwen3-0.6B
accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/a2d/bd3lm/sft.py \
    --model_name_or_path "Qwen/Qwen3-0.6B-Base" \
    --block_size 64

# Evaluate
bash examples/a2d/mdlm/eval.sh --model_name_or_path "outputs/model"
```

---

*Document version: 1.0 | Last updated: January 2026*
