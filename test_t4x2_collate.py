#!/usr/bin/env python3
"""
Test script to verify the T4x2 collate function works correctly
"""

import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import numpy as np

def collate_batch(batch):
    """Custom collate function for PyTorch Geometric Data objects."""
    # Filter out None items and ensure all items are Data objects
    valid_batch = []
    for item in batch:
        if item is not None and hasattr(item, 'x') and hasattr(item, 'edge_index'):
            # Ensure all items have y and mask attributes
            if not hasattr(item, 'y') or item.y is None:
                item.y = torch.zeros(5, dtype=torch.float)
            if not hasattr(item, 'mask') or item.mask is None:
                item.mask = torch.zeros(5, dtype=torch.float)
            valid_batch.append(item)
    
    # If no valid items, create a dummy batch
    if len(valid_batch) == 0:
        dummy_data = Data(x=torch.zeros((1, 32)), edge_index=torch.empty((2, 0), dtype=torch.long))
        dummy_data.y = torch.zeros(5, dtype=torch.float)
        dummy_data.mask = torch.zeros(5, dtype=torch.float)
        valid_batch = [dummy_data]
    
    # Create batch using PyTorch Geometric's Batch.from_data_list
    try:
        # First create the batch for graph structure
        batch_data = Batch.from_data_list(valid_batch)
        
        # Handle y and mask as graph-level attributes manually
        # Now all items should have y and mask attributes
        y_list = [item.y for item in valid_batch]
        mask_list = [item.mask for item in valid_batch]
        
        if y_list:
            batch_data.y = torch.stack(y_list, dim=0)  # Shape: (batch_size, num_properties)
        if mask_list:
            batch_data.mask = torch.stack(mask_list, dim=0)  # Shape: (batch_size, num_properties)
        
        return batch_data
        
    except Exception as e:
        print(f'Error in collate_batch: {e}')
        print(f'Batch sizes: {[item.y.shape if hasattr(item, "y") else "No y" for item in valid_batch[:3]]}')
        # Return a dummy batch as fallback
        dummy_data = Data(x=torch.zeros((1, 32)), edge_index=torch.empty((2, 0), dtype=torch.long))
        dummy_data.y = torch.zeros((1, 5))
        dummy_data.mask = torch.zeros((1, 5))
        return dummy_data

def create_test_data():
    """Create test data with mixed y and mask availability."""
    data_list = []
    
    for i in range(10):
        # Create random graph data
        num_nodes = np.random.randint(3, 8)
        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        
        data = Data(x=x, edge_index=edge_index)
        
        # Some items have targets, some don't (like supplementary data)
        if i % 3 == 0:  # 1/3 have no targets (like dataset2)
            # No y and mask attributes
            pass
        elif i % 3 == 1:  # 1/3 have partial targets
            y = torch.randn(5)
            mask = torch.rand(5) > 0.5  # Random mask
            data.y = y
            data.mask = mask.float()
        else:  # 1/3 have full targets
            y = torch.randn(5)
            mask = torch.ones(5)
            data.y = y
            data.mask = mask
        
        data_list.append(data)
    
    return data_list

def test_collate_function():
    """Test the collate function with mixed data."""
    print("🧪 Testing T4x2 collate function...")
    
    # Create test dataset
    dataset = create_test_data()
    
    # Test different batch sizes
    batch_sizes = [16, 32, 64]
    
    for batch_size in batch_sizes:
        print(f"\n🔍 Testing batch_size={batch_size}...")
        
        # Create data loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
        
        # Test one batch
        try:
            for batch_data in loader:
                print(f"   Batch info:")
                print(f"     x shape: {batch_data.x.shape}")
                print(f"     batch shape: {batch_data.batch.shape}")
                print(f"     y shape: {batch_data.y.shape}")
                print(f"     mask shape: {batch_data.mask.shape}")
                
                # Check shapes match
                batch_size_actual = batch_data.batch.max().item() + 1
                expected_y_shape = (batch_size_actual, 5)
                expected_mask_shape = (batch_size_actual, 5)
                
                if batch_data.y.shape != expected_y_shape:
                    print(f"   ❌ Y shape mismatch: got {batch_data.y.shape}, expected {expected_y_shape}")
                    return False
                
                if batch_data.mask.shape != expected_mask_shape:
                    print(f"   ❌ Mask shape mismatch: got {batch_data.mask.shape}, expected {expected_mask_shape}")
                    return False
                
                print(f"   ✅ Batch processed successfully - shapes match!")
                break  # Only test first batch
                
        except Exception as e:
            print(f"   ❌ Error processing batch: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    if test_collate_function():
        print("\n✅ T4x2 collate function test passed!")
        print("The tensor shape mismatch issue should now be resolved.")
    else:
        print("\n❌ T4x2 collate function test failed!")
        print("There are still issues with the collate function.")