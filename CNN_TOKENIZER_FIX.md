# CNN Tokenizer Fix Summary

## Issue Identified

**Error**: `RuntimeError: view size is not compatible with input tensor's size and stride`

**Root Cause**: After operations like pooling, permute, and convolutions, tensors can become non-contiguous in memory. The `.view()` method requires contiguous memory layout, which fails in these cases.

**Solution**: Replace all `.view()` calls with `.reshape()`, which automatically handles non-contiguous tensors by creating a contiguous copy when needed.

## Changes Made

### 1. StateCNNTokenizer (Encoder)
- **Line 117**: Changed `x.view(B, C * H * W)` → `x.reshape(B, C * H * W)`
- **Line 121**: Changed `x.view(B, self.num_tokens, self.hidden_dim)` → `x.reshape(B, self.num_tokens, self.hidden_dim)`

### 2. ActionCNNTokenizer (Decoder)
- **Line 196**: Changed `x.view(B, spatial_size, spatial_size, 128)` → `x.reshape(B, spatial_size, spatial_size, 128)`
- **Line 215**: Changed `x.reshape(B, num_tokens, self.num_actions)` to handle padding/cropping properly
- **Added**: Robust handling for cases where `num_tokens != spatial_size^2` with padding/cropping logic

## Technical Details

### Why `.view()` Fails
- `.view()` requires the tensor to be contiguous in memory
- Operations like `permute()`, pooling, and some convolutions can create non-contiguous tensors
- `.view()` throws an error when trying to reshape non-contiguous tensors

### Why `.reshape()` Works
- `.reshape()` automatically handles non-contiguous tensors
- If the tensor is contiguous, it behaves like `.view()` (no copy)
- If non-contiguous, it creates a contiguous copy first, then reshapes
- Slightly more memory overhead but more robust

## Testing

After this fix, the CNN architecture should work correctly:

```bash
python scripts/test_cnn_architecture.py
```

Expected output:
- ✓ StateEncoder test passes
- ✓ ActionEncoder test passes  
- ✓ ActionDecoder test passes
- ✓ End-to-end shape compatibility verified

## Impact

- **No breaking changes**: `.reshape()` is a drop-in replacement for `.view()` in these cases
- **More robust**: Handles edge cases with non-contiguous tensors
- **Slight performance**: Minimal overhead (only when tensors are non-contiguous)

## Files Modified

- `src/models/components/cnn_tokenizer.py`: All `.view()` calls replaced with `.reshape()`
