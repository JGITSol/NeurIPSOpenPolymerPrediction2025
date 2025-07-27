#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 Competition Script

This script handles training and prediction for the NeurIPS competition,
including proper data loading, multi-target prediction, and submission generation.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from datetime import datetime
import argparse

import sys
sys.path.append('src')

from polymer_prediction.config.config import CONFIG
from polymer_prediction.data.dataset import PolymerDataset
from polymer_prediction.models.gcn import PolymerGCN
from polymer_prediction.training.trainer import train_one_epoch, evaluate, predict
from polymer_prediction.utils.io import save_model


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
    """Load the competition data files.
    
    Returns:
        tuple: (train_df, test_df)
    """
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
    
    # Print data info
    print("\nTraining data columns:", train_df.columns.tolist())
    print("Training data shape:", train_df.shape)
    
    # Check for missing values in each target column
    target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    print("\nMissing values per target:")
    for col in target_cols:
        if col in train_df.columns:
            missing = train_df[col].isna().sum()
            total = len(train_df)
            print(f"  {col}: {missing}/{total} ({missing/total*100:.1f}%)")
    
    return train_df, test_df


def prepare_datasets(train_df, test_df, val_split=0.2):
    """Prepare datasets for training, validation, and testing.
    
    Args:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Test data
        val_split (float): Fraction of training data to use for validation
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
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


def train_model(train_dataset, val_dataset, config):
    """Train the model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration object
        
    Returns:
        tuple: (model, training_history)
    """
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
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_rmses': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, config.DEVICE)
        history['train_losses'].append(train_loss)
        
        # Validate
        val_loss, val_rmses = evaluate(model, val_loader, config.DEVICE)
        history['val_losses'].append(val_loss)
        history['val_rmses'].append(val_rmses)
        
        # Print progress
        rmse_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_rmses.items() if not np.isnan(v)])
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | RMSE: {rmse_str}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    
    return model, history


def generate_submission(model, test_dataset, config, output_path="submission.csv"):
    """Generate submission file for the competition.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        config: Configuration object
        output_path (str): Path to save submission file
    """
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
    parser = argparse.ArgumentParser(description="NeurIPS Open Polymer Prediction 2025")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_channels", type=int, default=128, help="Hidden channels")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GCN layers")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output submission file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG.NUM_EPOCHS = args.epochs
    CONFIG.BATCH_SIZE = args.batch_size
    CONFIG.LEARNING_RATE = args.lr
    CONFIG.HIDDEN_CHANNELS = args.hidden_channels
    CONFIG.NUM_GCN_LAYERS = args.num_layers
    CONFIG.SEED = args.seed
    
    # Set seed
    set_seed(CONFIG.SEED)
    
    # Create output directories
    os.makedirs(CONFIG.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(CONFIG.RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("NeurIPS Open Polymer Prediction 2025")
    print("=" * 60)
    
    # Load data
    train_df, test_df = load_competition_data()
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_df, test_df, val_split=args.val_split
    )
    
    # Train model
    model, history = train_model(train_dataset, val_dataset, CONFIG)
    
    # Generate submission
    submission_df = generate_submission(model, test_dataset, CONFIG, args.output)
    
    print("\n" + "=" * 60)
    print("Competition submission ready!")
    print(f"Submission file: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()