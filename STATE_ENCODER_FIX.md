# State Encoder Fix Summary

## Issue Identified

**Problem**: CNN tokenizer expected `[B, 7, 7, 3]` but received `[B, seq_len, 147]` (flattened grid from dataset).

**Error**: 
```
RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], 
expected input[1, 16, 64, 147] to have 3 channels, but got 16 channels instead
```

**Root Cause**: 
- Dataset stores states as sequences: `[B, seq_len, 147]` per sample
- StateEncoder was passing this directly to CNN tokenizer
- CNN tokenizer expects spatial format: `[B, 7, 7, 3]` or `[B, 3, 7, 7]`

## Fix Applied

### StateEncoder.forward() ✅

**Changes**:
1. **Handle sequence input**: If grid is `[B, seq_len, 147]`, take first state `grid[:, 0, :]` → `[B, 147]`
2. **Reshape flattened grid**: `[B, 147]` → `[B, 7, 7, 3]`
3. **Handle direction**: If direction is `[B, seq_len]`, take first `direction[:, 0]` → `[B]`
4. **Pass to CNN tokenizer**: Now gets correct format `[B, 7, 7, 3]`

### CNN Tokenizer ✅

**Changes**:
1. **Better format detection**: Check both `(H, W, C)` and `(C, H, W)` formats more robustly
2. **Clear error messages**: Better error reporting if format is unexpected

## Architecture Flow (Fixed)

```
Dataset Output:
  grid: [B, seq_len, 147]
  direction: [B, seq_len]

StateEncoder.forward():
  1. Take first state: grid[:, 0, :] → [B, 147]
  2. Reshape: [B, 147] → [B, 7, 7, 3]
  3. Take first direction: direction[:, 0] → [B]
  
CNN Tokenizer:
  Input: [B, 7, 7, 3]
  Convert to: [B, 3, 7, 7]
  CNN processing...
  Output: [B, 49, hidden_dim]

StateEncoder Output:
  [B, 49, hidden_dim] tokens
```

## Why First State Only?

For masked diffusion training:
- We condition on the **initial state** only
- The action sequence is denoised from scratch
- The initial state provides the maze context
- Subsequent states are not needed for conditioning

## Testing

After this fix:
1. Grid shape will be correctly reshaped from `[B, seq_len, 147]` → `[B, 7, 7, 3]`
2. Direction shape will be correctly extracted from `[B, seq_len]` → `[B]`
3. CNN tokenizer will receive correct input format
4. Training should proceed without shape errors

## Expected Behavior

```bash
python scripts/train_baseline.py
```

Should now:
- ✅ Reshape grid correctly
- ✅ Extract first state and direction
- ✅ Pass correct format to CNN
- ✅ Train without shape errors

## Files Modified

1. `src/models/state_encoder.py` - Added reshaping logic for sequence inputs
2. `src/models/components/cnn_tokenizer.py` - Better format detection

The fix maintains backward compatibility with single-state inputs while handling sequence inputs from the dataset.
