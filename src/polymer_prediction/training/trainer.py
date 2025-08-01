"""Training and evaluation utilities for polymer prediction models."""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def masked_mse_loss(predictions, targets, masks):
    """Calculate MSE loss only for non-missing values.
    
    Args:
        predictions (torch.Tensor): Model predictions (batch_size, 5)
        targets (torch.Tensor): Target values (batch_size, 5)
        masks (torch.Tensor): Mask indicating which values are present (batch_size, 5)
        
    Returns:
        torch.Tensor: Masked MSE loss
    """
    # Ensure all tensors have the same shape
    assert predictions.shape == targets.shape == masks.shape, f"Shape mismatch: pred {predictions.shape}, target {targets.shape}, mask {masks.shape}"
    
    # Only compute loss for non-missing values
    masked_predictions = predictions * masks
    masked_targets = targets * masks
    
    # Calculate squared differences
    squared_diff = (masked_predictions - masked_targets) ** 2
    
    # Sum over all dimensions and divide by number of non-missing values
    total_loss = torch.sum(squared_diff)
    total_count = torch.sum(masks)
    
    if total_count > 0:
        return total_loss / total_count
    else:
        return torch.tensor(0.0, device=predictions.device)


def train_one_epoch(model, loader, optimizer, device):
    """Perform one full training pass over the dataset.
    
    Args:
        model (torch.nn.Module): Model to train
        loader (torch_geometric.data.DataLoader): DataLoader for training data
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to use for training
        
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = masked_mse_loss(out, data.y, data.mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model on a dataset.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        loader (torch_geometric.data.DataLoader): DataLoader for evaluation data
        device (torch.device): Device to use for evaluation
        
    Returns:
        tuple: Average loss and per-property RMSE
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    all_preds = []
    all_targets = []
    all_masks = []
    
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = data.to(device)
        out = model(data)
        
        loss = masked_mse_loss(out, data.y, data.mask)
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        
        all_preds.append(out.cpu())
        all_targets.append(data.y.cpu())
        all_masks.append(data.mask.cpu())

    avg_loss = total_loss / total_samples
    
    # Calculate per-property RMSE
    preds = torch.cat(all_preds, dim=0)  # Shape: (N, 5)
    targets = torch.cat(all_targets, dim=0)  # Shape: (N, 5)
    masks = torch.cat(all_masks, dim=0)  # Shape: (N, 5)
    
    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    rmses = {}
    
    for i, prop_name in enumerate(property_names):
        prop_mask = masks[:, i]
        if prop_mask.sum() > 0:  # Only calculate if we have non-missing values
            prop_preds = preds[:, i][prop_mask == 1]
            prop_targets = targets[:, i][prop_mask == 1]
            rmse = torch.sqrt(torch.mean((prop_preds - prop_targets) ** 2))
            rmses[prop_name] = rmse.item()
        else:
            rmses[prop_name] = float('nan')
    
    return avg_loss, rmses


@torch.no_grad()
def predict(model, loader, device):
    """Generate predictions for test data.
    
    Args:
        model (torch.nn.Module): Trained model
        loader (torch_geometric.data.DataLoader): DataLoader for test data
        device (torch.device): Device to use for prediction
        
    Returns:
        tuple: (ids, predictions) where predictions is a numpy array of shape (N, 5)
    """
    model.eval()
    all_ids = []
    all_preds = []
    
    for data in tqdm(loader, desc="Predicting", leave=False):
        data = data.to(device)
        out = model(data)
        
        all_ids.extend(data.id)
        all_preds.append(out.cpu())
    
    predictions = torch.cat(all_preds, dim=0).numpy()
    
    return all_ids, predictions
