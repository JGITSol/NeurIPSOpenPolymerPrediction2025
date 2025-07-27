#!/usr/bin/env python3
"""
Final training script for NeurIPS Open Polymer Prediction 2025

This script trains the final model with optimized hyperparameters.
"""

import sys
sys.path.append('src')

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from datetime import datetime
import argparse

from polymer_prediction.config.config import CONFIG
from polymer_prediction.data.dataset import PolymerDataset
from polymer_prediction.models.gcn import PolymerGCN
from polymer_prediction.training.trainer import train_one_epoch, evaluate, predict
from polymer_prediction.utils.io import save_model
from polymer_prediction.utils.competition_metrics import weighted_mae, print_competition_metrics


def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_competition_data():
    """Load the competition data files."""
    train_path = "info/train.csv"
    test_path = "info/test.csv"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded training data: {len(train_df)} samples")
    print(f"Loaded test data: {len(test_df)} samples")
    
    return train_df, test_df


def prepare_datasets(train_df, test_df, val_split=0.15):
    """Prepare datasets for training, validation, and testing."""
    # Create datasets
    full_train_dataset = PolymerDataset(train_df, is_test=False)
    test_dataset = PolymerDataset(test_df, is_test=True)
    
    # Filter out invalid SMILES from training data
    valid_indices = []
    for i in range(len(full_train_dataset)):
        data = full_train_dataset.get(i)
        if data is not None:
            valid_indices.append(i)
    
    if len(valid_indices) != len(full_train_dataset):
        print(f"Warning: Filtered out {len(full_train_dataset) - len(valid_indices)} invalid SMILES from training data.")
    
    # Create clean training dataset
    train_df_clean = train_df.iloc[valid_indices].reset_index(drop=True)
    clean_train_dataset = PolymerDataset(train_df_clean, is_test=False)
    
    # Split training data into train and validation
    train_size = int((1.0 - val_split) * len(clean_train_dataset))
    val_size = len(clean_train_dataset) - train_size
    train_dataset, val_dataset = random_split(clean_train_dataset, [train_size, val_size])
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def train_model_with_scheduler(train_dataset, val_dataset, config):
    """Train the model with learning rate scheduling."""
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Get feature dimensions
    first_data = train_dataset[0]
    num_atom_features = first_data.num_atom_features
    
    # Initialize model
    model = PolymerGCN(
        num_atom_features=num_atom_features,
        hidden_channels=config.HIDDEN_CHANNELS,
        num_gcn_layers=config.NUM_GCN_LAYERS
    ).to(config.DEVICE)
    
    print(f"\nModel architecture:")
    print(f"  Input features: {num_atom_features}")
    print(f"  Hidden channels: {config.HIDDEN_CHANNELS}")
    print(f"  GCN layers: {config.NUM_GCN_LAYERS}")
    print(f"  Output targets: 5 (Tg, FFV, Tc, Density, Rg)")
    print(f"  Device: {config.DEVICE}")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_rmses': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20
    
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, config.DEVICE)
        history['train_losses'].append(train_loss)
        
        # Validate
        val_loss, val_rmses = evaluate(model, val_loader, config.DEVICE)
        history['val_losses'].append(val_loss)
        history['val_rmses'].append(val_rmses)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        rmse_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_rmses.items() if not np.isnan(v)])
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | RMSE: {rmse_str}")
        
        # Calculate competition metric every 10 epochs
        if epoch % 10 == 0:
            # Get validation predictions for competition metric
            val_ids, val_predictions = predict(model, val_loader, config.DEVICE)
            val_targets = []
            val_masks = []
            for data in val_loader:
                val_targets.append(data.y)
                val_masks.append(data.mask)
            val_targets = torch.cat(val_targets, dim=0).numpy()
            val_masks = torch.cat(val_masks, dim=0).numpy()
            
            wmae = weighted_mae(val_predictions, val_targets, val_masks)
            print(f"         Weighted MAE (Competition): {wmae:.6f}")
        
        # Save best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_metadata = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_rmses': val_rmses,
                'model_params': {
                    'num_atom_features': num_atom_features,
                    'hidden_channels': config.HIDDEN_CHANNELS,
                    'num_gcn_layers': config.NUM_GCN_LAYERS,
                }
            }
            save_model(model, os.path.join(config.MODEL_SAVE_DIR, 'best_model.pt'), model_metadata)
            print(f"  -> New best validation loss! Saved model.")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            break
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    
    return model, history


def generate_submission(model, test_dataset, config, output_path="submission.csv"):
    """Generate submission file for the competition."""
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"\nGenerating predictions for {len(test_dataset)} test samples...")
    
    # Get predictions
    ids, predictions = predict(model, test_loader, config.DEVICE)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': [int(id_val) if hasattr(id_val, 'item') else int(id_val) for id_val in ids],
        'Tg': predictions[:, 0],
        'FFV': predictions[:, 1],
        'Tc': predictions[:, 2],
        'Density': predictions[:, 3],
        'Rg': predictions[:, 4]
    })
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Print sample predictions
    print(f"\nSample predictions:")
    print(submission_df.head())
    
    return submission_df


def main():
    """Main function."""
    # Optimized hyperparameters
    CONFIG.NUM_EPOCHS = 100
    CONFIG.BATCH_SIZE = 64
    CONFIG.LEARNING_RATE = 0.001
    CONFIG.HIDDEN_CHANNELS = 256
    CONFIG.NUM_GCN_LAYERS = 4
    CONFIG.WEIGHT_DECAY = 1e-4
    CONFIG.SEED = 42
    
    # Set seed
    set_seed(CONFIG.SEED)
    
    # Create output directories
    os.makedirs(CONFIG.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(CONFIG.RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("NeurIPS Open Polymer Prediction 2025 - Final Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Epochs: {CONFIG.NUM_EPOCHS}")
    print(f"  Batch size: {CONFIG.BATCH_SIZE}")
    print(f"  Learning rate: {CONFIG.LEARNING_RATE}")
    print(f"  Hidden channels: {CONFIG.HIDDEN_CHANNELS}")
    print(f"  GCN layers: {CONFIG.NUM_GCN_LAYERS}")
    print(f"  Weight decay: {CONFIG.WEIGHT_DECAY}")
    print(f"  Device: {CONFIG.DEVICE}")
    
    # Load data
    train_df, test_df = load_competition_data()
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_df, test_df, val_split=0.15
    )
    
    # Train model
    model, history = train_model_with_scheduler(train_dataset, val_dataset, CONFIG)
    
    # Generate submission
    submission_df = generate_submission(model, test_dataset, CONFIG, "final_submission.csv")
    
    print("\n" + "=" * 60)
    print("Final model training complete!")
    print(f"Submission file: final_submission.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()