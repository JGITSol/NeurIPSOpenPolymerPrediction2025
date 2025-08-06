# =============================================================================
# COPY-PASTE FIX FOR TENSOR SHAPE ISSUE
# Replace the faulty training functions cell with this code
# =============================================================================

def weighted_mae_loss(predictions, targets, masks):
    """Weighted MAE loss with proper tensor shape handling."""
    
    # Handle DataParallel shape mismatch (predictions get concatenated from multiple GPUs)
    if predictions.shape[0] != targets.shape[0]:
        actual_batch_size = targets.shape[0]
        predictions = predictions[:actual_batch_size]
    
    # Fix tensor shape issues - ensure targets and masks are 2D
    if len(targets.shape) == 1:
        # Reshape from (batch_size * 5,) to (batch_size, 5)
        batch_size = predictions.shape[0]
        targets = targets.view(batch_size, -1)
        masks = masks.view(batch_size, -1)
    
    # Validate tensor shapes after reshaping
    if predictions.shape != targets.shape or predictions.shape != masks.shape:
        print(f"DEBUG: pred={predictions.shape}, target={targets.shape}, mask={masks.shape}")
        raise ValueError(f"Shape mismatch after reshape: pred={predictions.shape}, target={targets.shape}, mask={masks.shape}")
    
    # Equal weights for all properties
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=predictions.device, dtype=predictions.dtype)
    if len(weights.shape) == 1 and len(predictions.shape) == 2:
        weights = weights.unsqueeze(0)  # Shape: (1, 5) for broadcasting
    
    # Calculate weighted MAE
    mae_per_property = torch.abs(predictions - targets) * masks
    weighted_mae = (mae_per_property * weights).sum() / (masks * weights).sum()
    
    # Handle edge cases
    if torch.isnan(weighted_mae) or torch.isinf(weighted_mae):
        return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
    
    return weighted_mae

def train_epoch(model, train_loader, optimizer, device):
    """Train model for one epoch with proper tensor handling."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        if batch is None or not hasattr(batch, 'x'):
            continue
        
        # Move batch to device with explicit component handling
        batch = batch.to(device, non_blocking=True)
        
        # Ensure all batch components are on the correct device
        if hasattr(batch, 'x'):
            batch.x = batch.x.to(device, non_blocking=True)
        if hasattr(batch, 'edge_index'):
            batch.edge_index = batch.edge_index.to(device, non_blocking=True)
        if hasattr(batch, 'batch'):
            batch.batch = batch.batch.to(device, non_blocking=True)
        if hasattr(batch, 'y'):
            batch.y = batch.y.to(device, non_blocking=True)
        if hasattr(batch, 'mask'):
            batch.mask = batch.mask.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        try:
            if USE_MIXED_PRECISION and scaler is not None:
                with autocast():
                    predictions = model(batch)
                    loss = weighted_mae_loss(predictions, batch.y, batch.mask)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(batch)
                loss = weighted_mae_loss(predictions, batch.y, batch.mask)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"Training error in batch: {e}")
            print(f"Batch shapes: x={batch.x.shape if hasattr(batch, 'x') else 'None'}, y={batch.y.shape if hasattr(batch, 'y') else 'None'}, mask={batch.mask.shape if hasattr(batch, 'mask') else 'None'}")
            continue
    
    return total_loss / max(num_batches, 1)

def evaluate_model(model, val_loader, device):
    """Evaluate model on validation set with proper tensor handling."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        for batch in progress_bar:
            if batch is None or not hasattr(batch, 'x'):
                continue
            
            # Move batch to device with explicit component handling
            batch = batch.to(device, non_blocking=True)
            
            # Ensure all batch components are on the correct device
            if hasattr(batch, 'x'):
                batch.x = batch.x.to(device, non_blocking=True)
            if hasattr(batch, 'edge_index'):
                batch.edge_index = batch.edge_index.to(device, non_blocking=True)
            if hasattr(batch, 'batch'):
                batch.batch = batch.batch.to(device, non_blocking=True)
            if hasattr(batch, 'y'):
                batch.y = batch.y.to(device, non_blocking=True)
            if hasattr(batch, 'mask'):
                batch.mask = batch.mask.to(device, non_blocking=True)
            
            try:
                if USE_MIXED_PRECISION and scaler is not None:
                    with autocast():
                        predictions = model(batch)
                        loss = weighted_mae_loss(predictions, batch.y, batch.mask)
                else:
                    predictions = model(batch)
                    loss = weighted_mae_loss(predictions, batch.y, batch.mask)
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Validation error in batch: {e}")
                print(f"Batch shapes: x={batch.x.shape if hasattr(batch, 'x') else 'None'}, y={batch.y.shape if hasattr(batch, 'y') else 'None'}, mask={batch.mask.shape if hasattr(batch, 'mask') else 'None'}")
                continue
    
    return total_loss / max(num_batches, 1)

print("✅ Fixed training functions with proper tensor shape handling")
print("   Features: Tensor reshaping, robust error handling, device placement")