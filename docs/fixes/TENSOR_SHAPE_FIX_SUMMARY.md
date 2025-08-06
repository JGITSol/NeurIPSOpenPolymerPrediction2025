# Tensor Shape Mismatch Fix Summary

## Issue Identified
The Optuna optimization was failing with tensor shape mismatch errors:
```
⚠️ Trial failed: The size of tensor a (5) must match the size of tensor b (80) at non-singleton dimension 1
```

## Root Cause
PyTorch Geometric's `Batch.from_data_list()` treats `y` and `mask` attributes as node-level or edge-level attributes by default, causing them to be concatenated along the first dimension instead of being stacked as graph-level attributes.

### Problem Behavior:
- **Expected**: `y` shape `(batch_size, 5)`, `mask` shape `(batch_size, 5)`
- **Actual**: `y` shape `(batch_size * 5,)`, `mask` shape `(batch_size * 5,)`
- **Model Output**: `(batch_size, 5)`
- **Result**: Shape mismatch in loss calculation

## Solution Implemented

### Fixed Collate Function
Updated `collate_batch()` to manually handle `y` and `mask` as graph-level attributes:

```python
def collate_batch(batch):
    """Custom collate function for PyTorch Geometric Data objects."""
    # Filter valid batch items
    valid_batch = [item for item in batch 
                   if item is not None and hasattr(item, 'x') and hasattr(item, 'edge_index')]
    
    # Create batch for graph structure
    batch_data = Batch.from_data_list(valid_batch)
    
    # Handle y and mask as graph-level attributes manually
    if hasattr(valid_batch[0], 'y') and valid_batch[0].y is not None:
        y_list = [item.y for item in valid_batch if hasattr(item, 'y') and item.y is not None]
        if y_list:
            batch_data.y = torch.stack(y_list, dim=0)  # Shape: (batch_size, num_properties)
    
    if hasattr(valid_batch[0], 'mask') and valid_batch[0].mask is not None:
        mask_list = [item.mask for item in valid_batch if hasattr(item, 'mask') and item.mask is not None]
        if mask_list:
            batch_data.mask = torch.stack(mask_list, dim=0)  # Shape: (batch_size, num_properties)
    
    return batch_data
```

### Key Changes:
1. **Manual Stacking**: Use `torch.stack()` instead of letting PyTorch Geometric concatenate
2. **Graph-Level Attributes**: Ensure `y` and `mask` have shape `(batch_size, num_properties)`
3. **Proper Batching**: Maintain correct tensor dimensions for loss calculation

## Additional Improvements

### Enhanced Error Handling
Added comprehensive tensor shape validation:

```python
# Validate tensor shapes before loss calculation
if out.shape != data.y.shape or out.shape != data.mask.shape:
    raise ValueError(f"Tensor shape mismatch: out={out.shape}, y={data.y.shape}, mask={data.mask.shape}")
```

### Improved wmae_loss Function
Enhanced the loss function to handle broadcasting correctly:

```python
def wmae_loss(pred, target, mask, weights=None):
    if weights is None:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    # Ensure weights tensor has the right shape for broadcasting
    weights = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
    if len(weights.shape) == 1 and len(pred.shape) == 2:
        weights = weights.unsqueeze(0)  # Shape: (1, 5) for broadcasting
    
    diff = torch.abs(pred - target) * mask
    weighted_diff = diff * weights
    
    numerator = torch.sum(weighted_diff)
    denominator = torch.sum(mask * weights)
    
    return numerator / denominator if denominator > 0 else torch.tensor(0.0, device=pred.device)
```

### Better Debugging
Added comprehensive debugging information:
- Tensor shape logging
- Error context information
- Validation checks at multiple points

## Testing Results

### Before Fix:
```
y shape: torch.Size([80])      # Concatenated: 16 * 5 = 80
mask shape: torch.Size([80])   # Concatenated: 16 * 5 = 80
model output: torch.Size([16, 5])
❌ Shape mismatch error
```

### After Fix:
```
y shape: torch.Size([16, 5])   # Properly stacked
mask shape: torch.Size([16, 5]) # Properly stacked  
model output: torch.Size([16, 5])
✅ Shapes match correctly
```

## Impact on Optuna Optimization

### Before:
- All trials failing with tensor shape mismatch
- Optimization returning `inf` values
- No successful hyperparameter tuning

### After:
- Proper tensor shapes maintained
- Successful loss calculations
- Optuna optimization can proceed normally
- Expected performance improvements from hyperparameter tuning

## Files Modified

1. **cpu-only-optimized-v1-fixed.ipynb**:
   - Updated `collate_batch()` function
   - Enhanced `wmae_loss()` function
   - Added tensor shape validation
   - Improved error handling and debugging

2. **Test Files Created**:
   - `test_fixed_collate.py` - Validates the fix works correctly
   - `TENSOR_SHAPE_FIX_SUMMARY.md` - This documentation

## Verification

The fix has been thoroughly tested with different batch sizes (16, 32, 64) and confirms:
- ✅ Correct tensor shapes maintained
- ✅ Successful loss calculations
- ✅ No shape mismatch errors
- ✅ Ready for Optuna optimization

The notebook should now run Optuna hyperparameter optimization successfully without tensor shape mismatch errors.