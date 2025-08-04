#!/usr/bin/env python3
"""
Debug script to identify the tensor shape mismatch issue in Optuna optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import numpy as np

# Mock the components to test the issue
class MockPolymerGCN(nn.Module):
    def __init__(self, input_dim=15, hidden_channels=64, num_layers=3, output_dim=5, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, data):
        # Simple mock: just use the first node features
        x = data.x
        batch = data.batch
        
        # Get one feature per graph in the batch
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, 5)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.sum() > 0:
                # Use mean of node features for this graph
                graph_features = x[mask].mean(dim=0)[:15]  # Take first 15 features
                out[i] = self.linear(graph_features)
        
        return out

def wmae_loss(pred, target, mask, weights=None):
    """Weighted Mean Absolute Error loss."""
    if weights is None:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    # Ensure weights tensor has the right shape for broadcasting
    weights = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
    if len(weights.shape) == 1 and len(pred.shape) == 2:
        weights = weights.unsqueeze(0)  # Shape: (1, 5) for broadcasting with (batch_size, 5)
    
    diff = torch.abs(pred - target) * mask
    weighted_diff = diff * weights
    
    # Calculate weighted sum and normalization
    numerator = torch.sum(weighted_diff)
    denominator = torch.sum(mask * weights)
    
    # Avoid division by zero
    if denominator == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    return numerator / denominator

def collate_batch(batch):
    """Custom collate function for PyTorch Geometric Data objects."""
    # Filter out None items and ensure all items are Data objects
    valid_batch = []
    for item in batch:
        if item is not None and hasattr(item, 'x') and hasattr(item, 'edge_index'):
            valid_batch.append(item)
    
    # If no valid items, create a dummy batch
    if len(valid_batch) == 0:
        dummy_data = Data(x=torch.zeros((1, 15)), edge_index=torch.empty((2, 0), dtype=torch.long))
        dummy_data.y = torch.zeros(5)
        dummy_data.mask = torch.zeros(5)
        valid_batch = [dummy_data]
    
    # Create batch using PyTorch Geometric's Batch.from_data_list
    try:
        return Batch.from_data_list(valid_batch)
    except Exception as e:
        print(f'⚠️ Error in collate_batch: {e}')
        # Return a dummy batch as fallback
        dummy_data = Data(x=torch.zeros((1, 15)), edge_index=torch.empty((2, 0), dtype=torch.long))
        dummy_data.y = torch.zeros(5)
        dummy_data.mask = torch.zeros(5)
        return Batch.from_data_list([dummy_data])

def create_mock_dataset(size=100):
    """Create a mock dataset for testing."""
    data_list = []
    for i in range(size):
        # Create random graph data
        num_nodes = np.random.randint(3, 10)
        x = torch.randn(num_nodes, 15)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        
        # Create targets and mask
        y = torch.randn(5)
        mask = torch.ones(5)
        
        data = Data(x=x, edge_index=edge_index, y=y, mask=mask)
        data_list.append(data)
    
    return data_list

def test_batch_processing():
    """Test batch processing with different batch sizes."""
    print("🧪 Testing batch processing...")
    
    # Create mock dataset
    dataset = create_mock_dataset(50)
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64]
    
    for batch_size in batch_sizes:
        print(f"\n🔍 Testing batch_size={batch_size}...")
        
        # Create data loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        
        # Create model
        model = MockPolymerGCN(output_dim=5)
        
        # Test one batch
        try:
            for batch_data in loader:
                print(f"   Batch info:")
                print(f"     x shape: {batch_data.x.shape}")
                print(f"     batch shape: {batch_data.batch.shape}")
                print(f"     y shape: {batch_data.y.shape}")
                print(f"     mask shape: {batch_data.mask.shape}")
                
                # Forward pass
                out = model(batch_data)
                print(f"     model output shape: {out.shape}")
                
                # Check shapes match
                if out.shape != batch_data.y.shape:
                    print(f"   ❌ Shape mismatch: out={out.shape}, y={batch_data.y.shape}")
                    return False
                
                # Test loss calculation
                loss = wmae_loss(out, batch_data.y, batch_data.mask)
                print(f"     loss: {loss.item():.4f}")
                print(f"   ✅ Batch processed successfully")
                break  # Only test first batch
                
        except Exception as e:
            print(f"   ❌ Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    print("🔧 Debugging Optuna tensor shape issue...")
    
    if test_batch_processing():
        print("\n✅ All batch processing tests passed!")
        print("The issue might be in the actual model or data preparation.")
    else:
        print("\n❌ Batch processing tests failed!")
        print("Found the source of the tensor shape mismatch.")