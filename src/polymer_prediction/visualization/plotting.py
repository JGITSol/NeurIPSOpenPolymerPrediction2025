"""Plotting utilities for polymer prediction."""

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw


def plot_training_history(train_losses, val_losses, val_rmses, save_path=None):
    """Plot training history.
    
    Args:
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        val_rmses (list): List of validation RMSEs
        save_path (str, optional): Path to save the plot to
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot RMSE
    ax2.plot(epochs, val_rmses, 'g-', label='Validation RMSE')
    ax2.set_title('Validation RMSE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    
def plot_predictions(y_true, y_pred, save_path=None):
    """Plot true vs predicted values.
    
    Args:
        y_true (numpy.ndarray): True values
        y_pred (numpy.ndarray): Predicted values
        save_path (str, optional): Path to save the plot to
    """
    plt.figure(figsize=(8, 8))
    
    # Plot the scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot the perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    plt.title(f'True vs Predicted Values (RMSE: {rmse:.4f})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
    
    
def visualize_molecule(smiles, save_path=None):
    """Visualize a molecule from its SMILES representation.
    
    Args:
        smiles (str): SMILES representation of the molecule
        save_path (str, optional): Path to save the image to
        
    Returns:
        PIL.Image: Image of the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    
    img = Draw.MolToImage(mol, size=(300, 300))
    
    if save_path:
        img.save(save_path)
        
    return img
    
    
def visualize_molecules(smiles_list, labels=None, n_cols=3, save_path=None):
    """Visualize multiple molecules from their SMILES representations.
    
    Args:
        smiles_list (list): List of SMILES representations
        labels (list, optional): List of labels for each molecule
        n_cols (int, optional): Number of columns in the grid
        save_path (str, optional): Path to save the image to
        
    Returns:
        PIL.Image: Image of the molecules
    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    mols = [mol for mol in mols if mol is not None]
    
    if labels:
        labels = labels[:len(mols)]
    
    img = Draw.MolsToGridImage(mols, molsPerRow=n_cols, subImgSize=(200, 200), legends=labels)
    
    if save_path:
        img.save(save_path)
        
    return img
