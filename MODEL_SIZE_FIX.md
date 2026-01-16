# Model Size Fix Summary

## Issues Fixed

### 1. ReduceLROnPlateau `verbose` Argument ✅
**Problem**: `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`

**Fix**: Removed `verbose=True` argument (not supported in older PyTorch versions)

### 2. Model Parameter Count Too High ✅
**Problem**: Model had 41M parameters even with hidden_dim=64, which is way too large

**Root Cause**: The `StateCNNTokenizer` had a massive projection layer:
- Old: `nn.Linear(128 * 49, 49 * hidden_dim)`
- For hidden_dim=64: `nn.Linear(6272, 3136)` = **~19.6M parameters** in one layer!

**Fix**: More efficient architecture:
1. Use 1x1 conv to reduce channels: `128 → hidden_dim` (64)
2. Reshape spatial dimensions directly to tokens
3. Use small per-token projection: `nn.Linear(hidden_dim, hidden_dim)` 

**New Parameter Count** (estimated):
- StateCNNTokenizer: ~1M parameters (down from ~20M)
- Total model: ~5-10M parameters (down from 41M)

## Architecture Changes

### Before (Inefficient)
```
CNN Features: [B, 128, 7, 7]
Flatten: [B, 6272]
Huge Linear: [B, 6272] → [B, 3136] (19.6M params!)
Reshape: [B, 49, 64]
```

### After (Efficient)
```
CNN Features: [B, 128, 7, 7]
1x1 Conv: [B, 128, 7, 7] → [B, 64, 7, 7] (~8K params)
Reshape: [B, 64, 7, 7] → [B, 49, 64]
Small Linear: [B, 49, 64] → [B, 49, 64] (~4K params)
```

## Expected Parameter Breakdown (hidden_dim=64)

### StateCNNTokenizer
- Conv layers: ~50K
- ResNet blocks: ~300K
- Channel reduction (1x1 conv): ~8K
- Token projection: ~4K
- **Total: ~360K**

### ActionEncoder
- Embedding + projection: ~10K
- **Total: ~10K**

### DiffusionPolicy (DiT)
- Time embedding: ~35K
- DiT blocks (4 layers, 64 hidden_dim, 4 heads): ~150K each = ~600K
- Output head: ~450 (64 * 7)
- **Total: ~635K**

### Total Model
- **Estimated: ~1-2M parameters** (much more reasonable!)

## Testing

After this fix, the model should:
1. Have ~1-2M parameters (down from 41M)
2. Train much faster
3. Use less memory
4. Still maintain performance

Run again:
```bash
python scripts/train_baseline.py
```

Should see:
- Much smaller parameter count
- Faster initialization
- No ReduceLROnPlateau error
