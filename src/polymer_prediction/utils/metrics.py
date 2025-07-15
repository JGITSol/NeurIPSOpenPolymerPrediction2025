"""Evaluation metrics for polymer prediction models."""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Union, Tuple


def calculate_regression_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """Calculate comprehensive regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing various regression metrics
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Symmetric Mean Absolute Percentage Error
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    
    # Mean Bias Error
    mbe = np.mean(y_pred - y_true)
    
    # Normalized RMSE
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mape": float(mape),
        "smape": float(smape),
        "mbe": float(mbe),
        "nrmse": float(nrmse),
    }


def calculate_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate classification metrics for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities or logits
        threshold: Classification threshold
        
    Returns:
        Dictionary containing classification metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
    )
    
    # Convert to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Convert predictions to binary
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, average='binary')
    recall = recall_score(y_true, y_pred_binary, average='binary')
    f1 = f1_score(y_true, y_pred_binary, average='binary')
    
    # ROC AUC (requires probabilities)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def early_stopping_check(
    val_losses: list,
    patience: int = 10,
    min_delta: float = 1e-4,
) -> Tuple[bool, int]:
    """Check if early stopping criteria are met.
    
    Args:
        val_losses: List of validation losses
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        
    Returns:
        Tuple of (should_stop, epochs_without_improvement)
    """
    if len(val_losses) < patience + 1:
        return False, 0
    
    best_loss = min(val_losses[:-patience])
    recent_losses = val_losses[-patience:]
    
    # Check if any recent loss is better than best_loss - min_delta
    improved = any(loss < best_loss - min_delta for loss in recent_losses)
    
    if not improved:
        return True, patience
    
    # Count epochs without improvement
    epochs_without_improvement = 0
    for i in range(len(val_losses) - 1, 0, -1):
        if val_losses[i] >= val_losses[i-1] - min_delta:
            epochs_without_improvement += 1
        else:
            break
    
    return False, epochs_without_improvement