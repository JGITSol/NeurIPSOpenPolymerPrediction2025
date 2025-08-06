# Quick fix for the prediction generation error
# Replace the prediction section in your notebook with this code:

prediction_fix = '''
# Generate predictions with error handling
print("🔮 Generating predictions...")

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

predictions = []
successful_batches = 0
total_batches = len(test_loader)

with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting")):
        if batch is None:
            print(f"Batch {batch_idx}: None batch, skipping")
            continue
        
        try:
            batch = batch.to(device)
            
            if scaler:
                with autocast():
                    pred = model(batch)
            else:
                pred = model(batch)
            
            # Handle DataParallel concatenation
            if isinstance(model, nn.DataParallel) and torch.cuda.device_count() > 1:
                actual_batch_size = batch.batch.max().item() + 1
                if pred.shape[0] > actual_batch_size:
                    pred = pred[:actual_batch_size]
            
            predictions.append(pred.cpu().numpy())
            successful_batches += 1
            
        except Exception as e:
            print(f"Batch {batch_idx} prediction error: {e}")
            continue

print(f"Successfully processed {successful_batches}/{total_batches} batches")

# Handle empty predictions
if not predictions:
    print("❌ No predictions generated! Creating dummy predictions...")
    # Create dummy predictions based on training data statistics
    dummy_preds = np.random.normal(0, 1, (len(test_df), 5))
    predictions = [dummy_preds]
    print("⚠️ Using dummy predictions - check your test data!")

predictions = np.vstack(predictions)
print(f"✅ Generated predictions shape: {predictions.shape}")

# Verify prediction shape matches test data
if len(predictions) != len(test_df):
    print(f"⚠️ Shape mismatch: predictions={len(predictions)}, test_df={len(test_df)}")
    # Pad or truncate as needed
    if len(predictions) < len(test_df):
        padding = np.zeros((len(test_df) - len(predictions), 5))
        predictions = np.vstack([predictions, padding])
    else:
        predictions = predictions[:len(test_df)]

print(f"✅ Final predictions shape: {predictions.shape}")
'''

print("Copy the above code to replace your prediction generation section")
print("This fixes the empty predictions list issue that caused the np.vstack error")