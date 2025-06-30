"""Training and evaluation utilities for polymer prediction models."""

import numpy as np
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Perform one full training pass over the dataset.
    
    Args:
        model (torch.nn.Module): Model to train
        loader (torch_geometric.data.DataLoader): DataLoader for training data
        optimizer (torch.optim.Optimizer): Optimizer
        loss_fn (callable): Loss function
        device (torch.device): Device to use for training
        
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Evaluate the model on a dataset.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        loader (torch_geometric.data.DataLoader): DataLoader for evaluation data
        loss_fn (callable): Loss function
        device (torch.device): Device to use for evaluation
        
    Returns:
        tuple: Average loss and RMSE
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = data.to(device)
        out = model(data)
        loss = loss_fn(out, data.y)
        total_loss += loss.item() * data.num_graphs
        all_preds.append(out.cpu())
        all_targets.append(data.y.cpu())

    avg_loss = total_loss / len(loader.dataset)
    
    # Calculate RMSE
    preds = torch.cat(all_preds).numpy().flatten()
    targets = torch.cat(all_targets).numpy().flatten()
    rmse = np.sqrt(np.mean((preds - targets)**2))
    
    return avg_loss, rmse
