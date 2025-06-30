"""Main script for polymer property prediction."""

import argparse
import io
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader

from polymer_prediction.config.config import CONFIG
from polymer_prediction.data.dataset import PolymerDataset
from polymer_prediction.models.gcn import PolymerGCN
from polymer_prediction.training.trainer import train_one_epoch, evaluate
from polymer_prediction.utils.io import save_model, save_results
from polymer_prediction.visualization.plotting import plot_training_history, plot_predictions


def set_seed(seed):
    """Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(data_path=None):
    """Load data from file or create dummy data if not provided.
    
    Args:
        data_path (str, optional): Path to data file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        return pd.read_csv(data_path)
    else:
        print("Creating dummy data for demonstration")
        dummy_data = """
smiles,target_property
CC(C)c1ccccc1,152.2
c1ccccc1,80.1
CCO,-114.1
C=CC=C,-108.9
c1cnccc1,42.3
O=C(O)c1ccccc1,122.4
CN(C)C=O,-61.0
C1CCCCC1,6.5
*c1ccccc1*,100.0  
*CC(*)(C)C*,-50.0 
*OC(=O)c1ccc(C(=O)O)cc1*,350.0 
"""
        return pd.read_csv(io.StringIO(dummy_data))


def prepare_datasets(df, target_col, test_split_fraction):
    """Prepare datasets for training and testing.
    
    Args:
        df (pandas.DataFrame): DataFrame containing SMILES and target values
        target_col (str): Name of the column containing target values
        test_split_fraction (float): Fraction of data to use for testing
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Create dataset
    full_dataset = PolymerDataset(df, target_col=target_col)
    
    # Filter out any SMILES that failed to parse
    valid_indices = [i for i, data in enumerate(full_dataset) if data is not None]
    if len(valid_indices) != len(full_dataset):
        print(f"Warning: Filtered out {len(full_dataset) - len(valid_indices)} invalid SMILES.")
    
    # A cleaner way to handle valid data
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    clean_dataset = PolymerDataset(df_valid, target_col=target_col)
    
    # Split data
    train_size = int((1.0 - test_split_fraction) * len(clean_dataset))
    test_size = len(clean_dataset) - train_size
    train_dataset, test_dataset = random_split(clean_dataset, [train_size, test_size])
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def train_model(train_dataset, test_dataset, config):
    """Train a model on the given datasets.
    
    Args:
        train_dataset (torch_geometric.data.Dataset): Training dataset
        test_dataset (torch_geometric.data.Dataset): Testing dataset
        config (Config): Configuration object
        
    Returns:
        tuple: (model, training_history)
    """
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Get feature dimensions from the first data object
    first_data_point = train_dataset[0]
    num_atom_features = first_data_point.num_atom_features
    
    # Initialize model
    model = PolymerGCN(
        num_atom_features=num_atom_features,
        hidden_channels=config.HIDDEN_CHANNELS,
        num_gcn_layers=config.NUM_GCN_LAYERS
    ).to(config.DEVICE)
    
    print("\n--- Model Architecture ---")
    print(model)
    print("------------------------\n")
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = torch.nn.MSELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    val_rmses = []
    best_val_rmse = float('inf')
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, config.DEVICE)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        val_loss, val_rmse = evaluate(model, test_loader, loss_fn, config.DEVICE)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            model_metadata = {
                'epoch': epoch,
                'val_rmse': val_rmse,
                'model_params': {
                    'num_atom_features': num_atom_features,
                    'hidden_channels': config.HIDDEN_CHANNELS,
                    'num_gcn_layers': config.NUM_GCN_LAYERS,
                }
            }
            save_model(model, os.path.join(config.MODEL_SAVE_DIR, 'best_model.pt'), model_metadata)
            print(f"  -> New best validation RMSE! Saved model checkpoint.")
    
    print(f"\n--- Training Complete ---")
    print(f"Best Validation RMSE Achieved: {best_val_rmse:.4f}")
    print("-------------------------")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_rmses': val_rmses,
    }
    
    return model, history


def main(args):
    """Main function.
    
    Args:
        args (argparse.Namespace): Command-line arguments
    """
    # Set random seed for reproducibility
    set_seed(CONFIG.SEED)
    
    # Create output directories
    os.makedirs(CONFIG.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(CONFIG.RESULTS_DIR, exist_ok=True)
    
    # Load data
    df = load_data(args.data_path)
    print("--- Data Preview ---")
    print(df.head())
    print("-------------------")
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(
        df, 
        target_col=args.target_column, 
        test_split_fraction=CONFIG.TEST_SPLIT_FRACTION
    )
    
    # Train model
    model, history = train_model(train_dataset, test_dataset, CONFIG)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(CONFIG.RESULTS_DIR, f"training_history_{timestamp}.json")
    save_results(history, results_path)
    
    # Plot results if requested
    if args.plot:
        plot_training_history(
            history['train_losses'], 
            history['val_losses'], 
            history['val_rmses'],
            os.path.join(CONFIG.RESULTS_DIR, f"training_history_{timestamp}.png")
        )
    
    print(f"Results saved to {results_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymer Property Prediction")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data file")
    parser.add_argument("--target_column", type=str, default="target_property", help="Name of target column")
    parser.add_argument("--plot", action="store_true", help="Plot training history")
    
    args = parser.parse_args()
    main(args)
