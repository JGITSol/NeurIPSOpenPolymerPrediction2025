"""Competition-specific metrics for NeurIPS Open Polymer Prediction 2025."""

import numpy as np
import torch


def weighted_mae(predictions, targets, masks=None):
    """
    Calculate the weighted Mean Absolute Error (wMAE) as defined in the competition.
    
    The evaluation metric is defined as:
    wMAE = (1/N) * sum_i sum_j w_j * |y_pred_ij - y_true_ij|
    
    where w_j is the reweighting factor for property j:
    w_j = (5 / sqrt(N_j)) / R_j
    
    Args:
        predictions (np.ndarray or torch.Tensor): Predicted values, shape (N, 5)
        targets (np.ndarray or torch.Tensor): True values, shape (N, 5)
        masks (np.ndarray or torch.Tensor, optional): Mask for missing values, shape (N, 5)
        
    Returns:
        float: Weighted MAE score
    """
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    if masks is not None and torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    
    # If no masks provided, assume all values are present
    if masks is None:
        masks = np.ones_like(predictions)
    
    # Property names for reference
    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Calculate weights for each property
    weights = []
    total_samples = predictions.shape[0]
    
    for j in range(5):  # 5 properties
        # Count non-missing values for this property
        N_j = np.sum(masks[:, j])
        
        if N_j == 0:
            # If no samples for this property, weight is 0
            weights.append(0.0)
            continue
        
        # Calculate range R_j for this property (using only non-missing values)
        valid_mask = masks[:, j] == 1
        if np.sum(valid_mask) > 1:
            property_values = targets[valid_mask, j]
            R_j = np.max(property_values) - np.min(property_values)
            if R_j == 0:
                R_j = 1.0  # Avoid division by zero
        else:
            R_j = 1.0
        
        # Calculate weight: w_j = (5 / sqrt(N_j)) / R_j
        w_j = (5.0 / np.sqrt(N_j)) / R_j
        weights.append(w_j)
    
    # Normalize weights so they sum to 5
    weights = np.array(weights)
    if np.sum(weights) > 0:
        weights = weights * (5.0 / np.sum(weights))
    
    # Calculate weighted MAE
    total_weighted_error = 0.0
    total_count = 0
    
    for i in range(total_samples):
        for j in range(5):
            if masks[i, j] == 1:  # Only consider non-missing values
                error = abs(predictions[i, j] - targets[i, j])
                weighted_error = weights[j] * error
                total_weighted_error += weighted_error
                total_count += 1
    
    if total_count == 0:
        return float('inf')
    
    wmae = total_weighted_error / total_count
    
    return wmae


def calculate_property_metrics(predictions, targets, masks=None):
    """
    Calculate individual metrics for each property.
    
    Args:
        predictions (np.ndarray or torch.Tensor): Predicted values, shape (N, 5)
        targets (np.ndarray or torch.Tensor): True values, shape (N, 5)
        masks (np.ndarray or torch.Tensor, optional): Mask for missing values, shape (N, 5)
        
    Returns:
        dict: Dictionary with metrics for each property
    """
    # Convert to numpy if needed
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    if masks is not None and torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    
    # If no masks provided, assume all values are present
    if masks is None:
        masks = np.ones_like(predictions)
    
    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    metrics = {}
    
    for j, prop_name in enumerate(property_names):
        valid_mask = masks[:, j] == 1
        
        if np.sum(valid_mask) == 0:
            metrics[prop_name] = {
                'mae': float('nan'),
                'rmse': float('nan'),
                'count': 0,
                'range': float('nan')
            }
            continue
        
        pred_vals = predictions[valid_mask, j]
        true_vals = targets[valid_mask, j]
        
        mae = np.mean(np.abs(pred_vals - true_vals))
        rmse = np.sqrt(np.mean((pred_vals - true_vals) ** 2))
        count = len(pred_vals)
        value_range = np.max(true_vals) - np.min(true_vals) if len(true_vals) > 1 else 0.0
        
        metrics[prop_name] = {
            'mae': mae,
            'rmse': rmse,
            'count': count,
            'range': value_range
        }
    
    return metrics


def print_competition_metrics(predictions, targets, masks=None):
    """
    Print comprehensive metrics in a nice format.
    
    Args:
        predictions (np.ndarray or torch.Tensor): Predicted values, shape (N, 5)
        targets (np.ndarray or torch.Tensor): True values, shape (N, 5)
        masks (np.ndarray or torch.Tensor, optional): Mask for missing values, shape (N, 5)
    """
    wmae = weighted_mae(predictions, targets, masks)
    property_metrics = calculate_property_metrics(predictions, targets, masks)
    
    print("\n" + "=" * 60)
    print("COMPETITION METRICS")
    print("=" * 60)
    print(f"Weighted MAE (Competition Metric): {wmae:.6f}")
    print("\nPer-Property Metrics:")
    print("-" * 60)
    print(f"{'Property':<10} {'Count':<8} {'MAE':<12} {'RMSE':<12} {'Range':<12}")
    print("-" * 60)
    
    for prop_name, metrics in property_metrics.items():
        count = metrics['count']
        mae = metrics['mae']
        rmse = metrics['rmse']
        prop_range = metrics['range']
        
        if count > 0:
            print(f"{prop_name:<10} {count:<8} {mae:<12.6f} {rmse:<12.6f} {prop_range:<12.6f}")
        else:
            print(f"{prop_name:<10} {count:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print("=" * 60)