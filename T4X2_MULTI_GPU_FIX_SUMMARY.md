# T4 x2 Multi-GPU Device Placement Fix Summary

## Issues Fixed

The T4 x2 notebook was experiencing multi-GPU device placement errors with the following symptoms:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
```

## Root Causes Identified

1. **Incorrect DataParallel Setup Order**: Model was wrapped with DataParallel before being moved to the primary device
2. **Device Placement in Forward Method**: Model didn't ensure input tensors were on the correct device
3. **Collate Function Issues**: Similar tensor shape issues as in the CPU notebook
4. **Primary Device Specification**: Using generic 'cuda' instead of explicit 'cuda:0'

## Fixes Applied

### 1. Fixed DataParallel Setup Order
**Before (WRONG):**
```python
# Multi-GPU setup
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # Wrap BEFORE moving to device
model = model.to(device)  # Too late!
```

**After (CORRECT):**
```python
# Move model to primary device FIRST
model = model.to(device)

# Multi-GPU setup AFTER moving to device
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### 2. Enhanced Model Forward Method
**Added device placement in forward method:**
```python
def forward(self, data):
    # Ensure all tensors are on the same device as model parameters
    device = next(self.parameters()).device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    batch = data.batch.to(device)
    
    # Rest of forward pass...
```

### 3. Fixed Collate Function
**Updated collate function to handle y and mask as graph-level attributes:**
```python
def collate_batch(batch):
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

### 4. Explicit Primary Device
**Changed device specification:**
```python
# Before
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# After  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

### 5. Enhanced Loss Function
**Improved weighted_mae_loss with better broadcasting:**
```python
def weighted_mae_loss(predictions, targets, masks):
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=predictions.device, dtype=predictions.dtype)
    
    # Ensure proper broadcasting
    if len(weights.shape) == 1 and len(predictions.shape) == 2:
        weights = weights.unsqueeze(0)  # Shape: (1, 5) for broadcasting
    
    mae_per_property = torch.abs(predictions - targets) * masks
    weighted_mae = (mae_per_property * weights).sum() / (masks * weights).sum()
    
    # Avoid division by zero
    if torch.isnan(weighted_mae) or torch.isinf(weighted_mae):
        return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
    
    return weighted_mae
```

### 6. Added Error Handling and Debugging
**Enhanced training and evaluation functions with device error detection:**
```python
try:
    if USE_MIXED_PRECISION:
        with autocast():
            predictions = model(batch)
            loss = weighted_mae_loss(predictions, batch.y, batch.mask)
    else:
        predictions = model(batch)
        loss = weighted_mae_loss(predictions, batch.y, batch.mask)
except RuntimeError as e:
    if "Expected all tensors to be on the same device" in str(e):
        print(f"Device error: {e}")
        print(f"Batch device: {batch.x.device if hasattr(batch, 'x') else 'N/A'}")
        print(f"Model device: {next(model.parameters()).device}")
    raise e
```

## Key Principles for Multi-GPU Setup

1. **Device Order**: Always move model to primary device BEFORE DataParallel wrapping
2. **Primary Device**: Use explicit device specification (cuda:0) for clarity
3. **Input Handling**: Ensure all inputs are moved to primary device in training loop
4. **Forward Method**: Add device placement checks in model forward method
5. **Error Handling**: Include device debugging for troubleshooting

## Expected Results

After these fixes, the T4 x2 notebook should:
- ✅ Run without device placement errors
- ✅ Properly utilize both GPUs with DataParallel
- ✅ Handle tensor shapes correctly in loss calculations
- ✅ Complete training successfully with mixed precision
- ✅ Generate valid predictions for submission

## Files Modified

- `neurips-t4x2-complete-solution-fixed.ipynb` - Applied all multi-GPU fixes
- `T4X2_MULTI_GPU_FIX_SUMMARY.md` - This documentation

The notebook is now ready for T4 x2 GPU training without device placement issues.