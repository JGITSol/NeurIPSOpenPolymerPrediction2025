"""Tests for GCN model training functionality."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch

# Mock GCN model for testing
class MockPolymerGCN(nn.Module):
    def __init__(self, num_atom_features, hidden_channels=128, num_targets=5):
        super().__init__()
        self.num_atom_features = num_atom_features
        self.hidden_channels = hidden_channels
        self.num_targets = num_targets
        self.linear = nn.Linear(num_atom_features, num_targets)
    
    def forward(self, data):
        x = data.x.mean(dim=0, keepdim=True)  # Simple aggregation
        return self.linear(x)

def test_gcn_model_initialization():
    """Test GCN model initialization."""
    model = MockPolymerGCN(num_atom_features=10, hidden_channels=64, num_targets=5)
    assert model.num_atom_features == 10
    assert model.hidden_channels == 64
    assert model.num_targets == 5

def test_gcn_forward_pass():
    """Test GCN model forward pass."""
    model = MockPolymerGCN(num_atom_features=10, num_targets=5)
    
    # Create sample data
    data = Data(
        x=torch.randn(5, 10),  # 5 nodes, 10 features each
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
    )
    
    # Forward pass
    output = model(data)
    
    # Check output shape
    assert output.shape == (1, 5)  # 1 graph, 5 targets
    assert not torch.isnan(output).any()

def test_masked_mse_loss():
    """Test masked MSE loss function."""
    def masked_mse_loss(predictions, targets, masks):
        valid_mask = masks.bool()
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        valid_predictions = predictions[valid_mask]
        valid_targets = targets[valid_mask]
        return nn.MSELoss()(valid_predictions, valid_targets)
    
    predictions = torch.randn(2, 3)
    targets = torch.randn(2, 3)
    masks = torch.ones(2, 3)
    
    loss = masked_mse_loss(predictions, targets, masks)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.item() >= 0

def test_training_step():
    """Test a single training step."""
    model = MockPolymerGCN(num_atom_features=10, num_targets=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create sample data
    data = Data(
        x=torch.randn(5, 10),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
        y=torch.randn(1, 5),
        mask=torch.ones(1, 5)
    )
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    predictions = model(data)
    loss = nn.MSELoss()(predictions, data.y)
    
    loss.backward()
    optimizer.step()
    
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0

def test_model_evaluation():
    """Test model evaluation."""
    model = MockPolymerGCN(num_atom_features=10, num_targets=5)
    
    # Create sample data
    data = Data(
        x=torch.randn(5, 10),
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
        y=torch.randn(1, 5),
        mask=torch.ones(1, 5)
    )
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        loss = nn.MSELoss()(predictions, data.y)
    
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0
    assert predictions.shape == data.y.shape