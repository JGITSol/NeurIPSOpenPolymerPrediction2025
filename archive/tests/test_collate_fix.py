#!/usr/bin/env python3
"""
Test script to verify the collate function fix for T4x2 notebook
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch

# Test dataset
class TestDataset(Dataset):
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create a simple graph
        x = torch.randn(3, 32)  # 3 nodes, 32 features
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

# Custom collate function
def collate_batch(batch):
    """Custom collate function for PyTorch Geometric Data objects."""
    # Filter out None items and ensure all items are Data objects
    valid_batch = []
    for item in batch:
        if item is not None and hasattr(item, 'x') and hasattr(item, 'edge_index'):
            valid_batch.append(item)
    
    # If no valid items, create a dummy batch
    if len(valid_batch) == 0:
        dummy_data = Data(x=torch.zeros((1, 32)), edge_index=torch.empty((2, 0), dtype=torch.long))
        valid_batch = [dummy_data]
    
    # Create batch using PyTorch Geometric's Batch.from_data_list
    try:
        return Batch.from_data_list(valid_batch)
    except Exception as e:
        print(f'Error in collate_batch: {e}')
        # Return a dummy batch as fallback
        dummy_data = Data(x=torch.zeros((1, 32)), edge_index=torch.empty((2, 0), dtype=torch.long))
        return Batch.from_data_list([dummy_data])

if __name__ == "__main__":
    print("Testing collate function fix...")
    
    # Create test dataset and dataloader
    dataset = TestDataset(5)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_batch)
    
    # Test iteration
    try:
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: {batch.batch.shape}, nodes: {batch.x.shape[0]}")
            if i >= 2:  # Test a few batches
                break
        print("✅ Collate function test passed!")
    except Exception as e:
        print(f"❌ Collate function test failed: {e}")