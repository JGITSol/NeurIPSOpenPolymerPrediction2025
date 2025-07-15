"""Tests for the metrics module."""

import numpy as np
import pytest
import torch

from polymer_prediction.utils.metrics import (
    calculate_regression_metrics,
    calculate_classification_metrics,
    early_stopping_check,
)


def test_calculate_regression_metrics():
    """Test regression metrics calculation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
    
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    # Check that all expected metrics are present
    expected_metrics = ['mse', 'rmse', 'mae', 'r2', 'mape', 'smape', 'mbe', 'nrmse']
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], float)
    
    # Check that RMSE is sqrt of MSE
    assert abs(metrics['rmse'] - np.sqrt(metrics['mse'])) < 1e-6
    
    # Check that R2 is reasonable (should be close to 1 for good predictions)
    assert metrics['r2'] > 0.9


def test_calculate_regression_metrics_torch():
    """Test regression metrics with PyTorch tensors."""
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = torch.tensor([1.1, 2.1, 2.9, 3.8, 5.2])
    
    metrics = calculate_regression_metrics(y_true, y_pred)
    
    # Should work the same as with numpy arrays
    assert 'rmse' in metrics
    assert isinstance(metrics['rmse'], float)


def test_calculate_classification_metrics():
    """Test classification metrics calculation."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
    
    metrics = calculate_classification_metrics(y_true, y_pred)
    
    # Check that all expected metrics are present
    expected_metrics = [
        'accuracy', 'precision', 'recall', 'f1', 'auc',
        'true_negatives', 'false_positives', 'false_negatives', 'true_positives'
    ]
    for metric in expected_metrics:
        assert metric in metrics
    
    # Check that confusion matrix values are integers
    assert isinstance(metrics['true_positives'], int)
    assert isinstance(metrics['false_positives'], int)
    
    # Check that AUC is reasonable
    assert 0 <= metrics['auc'] <= 1


def test_early_stopping_check():
    """Test early stopping logic."""
    # Case 1: Not enough epochs
    val_losses = [1.0, 0.9, 0.8]
    should_stop, epochs_without_improvement = early_stopping_check(val_losses, patience=5)
    assert should_stop is False
    
    # Case 2: Improving losses
    val_losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    should_stop, epochs_without_improvement = early_stopping_check(val_losses, patience=5)
    assert should_stop is False
    
    # Case 3: No improvement for patience epochs
    val_losses = [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    should_stop, epochs_without_improvement = early_stopping_check(val_losses, patience=5)
    assert should_stop is True
    assert epochs_without_improvement == 5


def test_early_stopping_check_with_min_delta():
    """Test early stopping with minimum delta."""
    # Small improvements that don't meet min_delta threshold
    val_losses = [1.0, 0.999, 0.998, 0.997, 0.996, 0.995, 0.994, 0.993, 0.992, 0.991, 0.990]
    should_stop, epochs_without_improvement = early_stopping_check(
        val_losses, patience=5, min_delta=0.01
    )
    assert should_stop is True