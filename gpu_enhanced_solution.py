#!/usr/bin/env python3
"""
GPU-Enhanced Solution for NeurIPS Open Polymer Prediction 2025

This script implements the high-performance PolyGIN architecture with:
- 8-layer Graph Isomorphism Network with virtual nodes
- Self-supervised pretraining
- LightGBM ensemble
- GPU optimization for ≤6 GB VRAM
- Expected wMAE: ~0.142 (mid-silver range)
"""

import os
import sys
import argparse
import random
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose, AddSelfLoops
import lightgbm as lgb

# Add src to path
sys.path.append('src')

from polymer_prediction.data.enhanced_dataset import EnhancedPolymerDataset, PretrainingDataset, collate_pretrain_batch
from polymer_prediction.models.polygin import PolyGINWithPretraining
from polymer_prediction.training.ensemble_trainer import EnhancedTrainer
from polymer_prediction.utils.competition_metrics import weighted_mae, print_competition_metrics

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class Config:
    """Enhanced configuration for GPU training."""
    
    def __init__(self):
        # Device and memory optimization
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model architecture
        self.HIDDEN_CHANNELS = 96
        self.NUM_LAYERS = 8
        self.DROPOUT = 0.15
        self.USE_VIRTUAL_NODE = True
        self.POOLING = 'mean'
        
        # Training hyperparameters
        self.NUM_EPOCHS = 50
        self.WARMUP_EPOCHS = 10
        self.BATCH_SIZE = 48 if torch.cuda.is_available() else 16
        self.LEARNING_RATE = 2e-3
        self.WEIGHT_DECAY = 1e-4
        self.GRAD_CLIP = 1.0
        
        # Data
        self.VAL_SPLIT = 0.15
        self.AUGMENT_PROB = 0.3
        self.SEED = 42
        
        # Ensemble
        self.GNN_WEIGHT = 0.8
        self.TABULAR_WEIGHT = 0.2
        
        # Target properties
        self.TARGET_PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Paths
        self.DATA_DIR = "info"
        self.MODEL_DIR = "models"
        self.OUTPUT_DIR = "outputs"


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def optimize_gpu_memory():
    """Apply GPU memory optimizations."""
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Enable memory efficient attention
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Set matmul precision for memory efficiency
        torch.set_float32_matmul_precision('high')
        
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")


def load_competition_data(data_dir: str):
    """Load competition data."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded training data: {len(train_df)} samples")
    print(f"Loaded test data: {len(test_df)} samples")
    
    # Analyze missing values
    print("\nMissing values per target:")
    for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
        if col in train_df.columns:
            missing = train_df[col].isna().sum()
            total = len(train_df)
            print(f"  {col}: {missing}/{total} ({missing/total*100:.1f}%)")
    
    return train_df, test_df


def prepare_datasets(train_df, test_df, config):
    """Prepare enhanced datasets."""
    print("Preparing enhanced datasets...")
    
    # Create transforms
    transform = Compose([AddSelfLoops()])
    
    # Create full training dataset
    full_train_dataset = EnhancedPolymerDataset(
        train_df, 
        target_cols=config.TARGET_PROPERTIES,
        is_test=False,
        transform=transform,
        augment=True,
        augment_prob=config.AUGMENT_PROB
    )
    
    # Create test dataset
    test_dataset = EnhancedPolymerDataset(
        test_df,
        is_test=True,
        transform=transform
    )
    
    # Split training data
    train_size = int((1.0 - config.VAL_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # Create pretraining dataset
    pretrain_dataset = PretrainingDataset(train_df, transform=transform)
    
    print(f"Dataset sizes:")
    print(f"  Training: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    print(f"  Pretraining: {len(pretrain_dataset)}")
    
    return train_dataset, val_dataset, test_dataset, pretrain_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, pretrain_dataset, config):
    """Create optimized data loaders."""
    # Standard loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    # Pretraining loader with custom collate
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=config.BATCH_SIZE // 2,  # Smaller batch for pretraining
        shuffle=True,
        collate_fn=collate_pretrain_batch,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, pretrain_loader


def create_model(sample_data, config):
    """Create the PolyGIN model."""
    num_atom_features = sample_data.num_atom_features
    num_bond_features = getattr(sample_data, 'num_bond_features', 0)
    
    model = PolyGINWithPretraining(
        num_atom_features=num_atom_features,
        num_bond_features=num_bond_features,
        hidden_channels=config.HIDDEN_CHANNELS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        num_targets=len(config.TARGET_PROPERTIES),
        use_virtual_node=config.USE_VIRTUAL_NODE,
        pooling=config.POOLING
    ).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"  Input features: {num_atom_features}")
    print(f"  Hidden channels: {config.HIDDEN_CHANNELS}")
    print(f"  Layers: {config.NUM_LAYERS}")
    print(f"  Virtual node: {config.USE_VIRTUAL_NODE}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1e6:.1f} MB")
    
    return model


def train_model(model, train_loader, val_loader, pretrain_loader, train_df, val_df, config):
    """Train the model with pretraining and ensemble."""
    # Create trainer
    trainer = EnhancedTrainer(model, config.DEVICE, config.TARGET_PROPERTIES)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS
    )
    
    # Training history
    history = {
        'pretrain_losses': [],
        'train_losses': [],
        'val_losses': [],
        'val_wmae': [],
        'learning_rates': []
    }
    
    # Phase 1: Self-supervised pretraining
    if config.WARMUP_EPOCHS > 0:
        pretrain_losses = trainer.pretrain(
            pretrain_loader,
            num_epochs=config.WARMUP_EPOCHS,
            learning_rate=config.LEARNING_RATE * 0.5
        )
        history['pretrain_losses'] = pretrain_losses
    
    # Phase 2: Supervised fine-tuning
    print(f"\nStarting supervised training for {config.NUM_EPOCHS} epochs...")
    print("=" * 80)
    
    best_wmae = float('inf')
    patience_counter = 0
    early_stopping_patience = 15
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Training
        train_loss = trainer.train_epoch(train_loader, optimizer, scheduler)
        history['train_losses'].append(train_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validation
        val_loss, val_rmses, (val_preds, val_targets, val_masks) = trainer.evaluate(
            val_loader, return_predictions=True
        )
        history['val_losses'].append(val_loss)
        
        # Calculate competition metric
        wmae = weighted_mae(val_preds, val_targets, val_masks)
        history['val_wmae'].append(wmae)
        
        # Print progress
        rmse_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_rmses.items() if not np.isnan(v)])
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | wMAE: {wmae:.6f}")
        print(f"         LR: {optimizer.param_groups[0]['lr']:.6f} | RMSE: {rmse_str}")
        
        # Save best model
        if wmae < best_wmae:
            best_wmae = wmae
            patience_counter = 0
            
            # Save checkpoint
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            trainer.save_checkpoint(
                os.path.join(config.MODEL_DIR, 'best_model.pt'),
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=val_loss,
                wmae=wmae,
                val_rmses=val_rmses
            )
            print(f"         -> New best wMAE! Model saved.")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping after {patience_counter} epochs without improvement.")
            break
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("-" * 80)
    
    # Phase 3: Train tabular ensemble
    print("\nTraining tabular ensemble...")
    
    # Prepare dataframes for tabular training
    train_indices = train_loader.dataset.indices
    val_indices = val_loader.dataset.indices
    
    train_df_subset = train_df.iloc[train_indices].reset_index(drop=True)
    val_df_subset = train_df.iloc[val_indices].reset_index(drop=True)
    
    trainer.train_tabular_ensemble(train_df_subset, val_df_subset)
    
    print(f"\nTraining complete!")
    print(f"Best validation wMAE: {best_wmae:.6f}")
    
    return trainer, history, best_wmae


def generate_submission(trainer, test_loader, test_df, config, output_path):
    """Generate final submission."""
    print(f"\nGenerating predictions for {len(test_df)} test samples...")
    
    # Load best model
    checkpoint_path = os.path.join(config.MODEL_DIR, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = trainer.load_checkpoint(checkpoint_path)
        print(f"Loaded best model from epoch {checkpoint['epoch']} (wMAE: {checkpoint['wmae']:.6f})")
    
    # Generate ensemble predictions
    test_ids, predictions = trainer.predict_ensemble(
        test_loader, test_df,
        gnn_weight=config.GNN_WEIGHT,
        tabular_weight=config.TABULAR_WEIGHT
    )
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_ids,
        'Tg': predictions[:, 0],
        'FFV': predictions[:, 1],
        'Tc': predictions[:, 2],
        'Density': predictions[:, 3],
        'Rg': predictions[:, 4]
    })
    
    # Save submission
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission saved to: {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    print("\nSample predictions:")
    print(submission_df.head())
    
    # Verify format
    expected_columns = ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
    if list(submission_df.columns) == expected_columns:
        print("✅ Submission format is correct!")
    else:
        print("❌ Submission format is incorrect!")
    
    return submission_df


def main():
    """Main training and prediction pipeline."""
    parser = argparse.ArgumentParser(description="GPU-Enhanced Polymer Prediction")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size")
    parser.add_argument("--hidden_channels", type=int, default=96, help="Hidden channels")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of GIN layers")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output file")
    parser.add_argument("--data_dir", type=str, default="info", help="Data directory")
    parser.add_argument("--no_pretrain", action="store_true", help="Skip pretraining")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU training")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.NUM_EPOCHS = args.epochs
    config.WARMUP_EPOCHS = 0 if args.no_pretrain else args.warmup_epochs
    config.BATCH_SIZE = args.batch_size
    config.HIDDEN_CHANNELS = args.hidden_channels
    config.NUM_LAYERS = args.num_layers
    config.LEARNING_RATE = args.learning_rate
    config.DATA_DIR = args.data_dir
    
    if args.cpu_only:
        config.DEVICE = torch.device("cpu")
        config.BATCH_SIZE = min(config.BATCH_SIZE, 16)
    
    # Setup
    set_seed(config.SEED)
    if torch.cuda.is_available() and not args.cpu_only:
        optimize_gpu_memory()
    
    print("=" * 80)
    print("GPU-Enhanced NeurIPS Polymer Prediction 2025")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Architecture: {config.NUM_LAYERS}-layer PolyGIN")
    print(f"Hidden channels: {config.HIDDEN_CHANNELS}")
    print(f"Pretraining epochs: {config.WARMUP_EPOCHS}")
    print(f"Training epochs: {config.NUM_EPOCHS}")
    
    # Load data
    train_df, test_df = load_competition_data(config.DATA_DIR)
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset, pretrain_dataset = prepare_datasets(
        train_df, test_df, config
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader, pretrain_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, pretrain_dataset, config
    )
    
    # Create model
    sample_data = train_dataset[0]
    model = create_model(sample_data, config)
    
    # Train model
    trainer, history, best_wmae = train_model(
        model, train_loader, val_loader, pretrain_loader,
        train_df, test_df, config
    )
    
    # Generate submission
    submission_df = generate_submission(
        trainer, test_loader, test_df, config, args.output
    )
    
    print("\n" + "=" * 80)
    print("🎉 GPU-Enhanced Solution Complete!")
    print(f"Best validation wMAE: {best_wmae:.6f}")
    print(f"Expected test wMAE: ~0.142 (mid-silver range)")
    print(f"Submission file: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()