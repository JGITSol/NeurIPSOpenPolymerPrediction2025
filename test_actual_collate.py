#!/usr/bin/env python3
"""
Test the actual collate function with realistic data that mimics the enhanced dataset
"""

import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# Mock the enhanced dataset behavior
class MockEnhancedDataset:
    def __init__(self, size=100, target_columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
        self.target_columns = target_columns
        self.data = []
        
        for i in range(size):
            # Create random graph data
            num_nodes = np.random.randint(3, 8)
            x = torch.randn(num_nodes, 32)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            
            data = Data(x=x, edge_index=edge_index)
            
            # Simulate enhanced dataset behavior with mixed target availability
            targets = []
            masks = []
            
            if self.target_columns:
                for col in self.target_columns:
                    # Simulate missing values like in supplementary data
                    if np.random.random() > 0.3:  # 70% chance of having value
                        targets.append(np.random.randn())
                        masks.append(1.0)
                    else:
                        targets.append(0.0)
                        masks.append(0.0)
            else:
                # For test data or data without targets
                targets = [0.0] * 5
                masks = [0.0] * 5
            
            data.y = torch.tensor(targets, dtype=torch.float)
            data.mask = torch.tensor(masks, dtype=torch.float)
            
            self.data.append(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_batch(batch):
    """Exact copy of the notebook's collate function."""
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
        # Store y and mask before creating batch to prevent PyG from concatenating them
        y_list = [item.y.clone() for item in valid_batch]
        mask_list = [item.mask.clone() for item in valid_batch]
        
        # Temporarily remove y and mask from items to prevent PyG concatenation
        for item in valid_batch:
            if hasattr(item, 'y'):
                delattr(item, 'y')
            if hasattr(item, 'mask'):
                delattr(item, 'mask')
        
        # Create the batch for graph structure only
        batch_data = Batch.from_data_list(valid_batch)
        
        # Add y and mask as graph-level attributes manually
        if y_list:
            batch_data.y = torch.stack(y_list, dim=0)  # Shape: (batch_size, num_properties)
        if mask_list:
            batch_data.mask = torch.stack(mask_list, dim=0)  # Shape: (batch_size, num_properties)
        
        # Debug: Check final shapes
        batch_size = batch_data.batch.max().item() + 1
        if hasattr(batch_data, 'y') and batch_data.y.shape[0] != batch_size:
            print(f"⚠️ Collate shape mismatch: batch_size={batch_size}, y.shape={batch_data.y.shape}")
            print(f"   y_list lengths: {[y.shape for y in y_list[:3]]}")
        
        return batch_data
        
    except Exception as e:
        print(f'Error in collate_batch: {e}')
        print(f'Batch sizes: {[item.y.shape if hasattr(item, "y") else "No y" for item in valid_batch[:3]]}')
        # Return a dummy batch as fallback
        dummy_data = Data(x=torch.zeros((1, 32)), edge_index=torch.empty((2, 0), dtype=torch.long))
        dummy_data.y = torch.zeros((1, 5))
        dummy_data.mask = torch.zeros((1, 5))
        return dummy_data

def test_realistic_collate():
    """Test with realistic enhanced dataset."""
    print("🧪 Testing realistic enhanced dataset collate...")
    
    # Create datasets similar to the notebook
    train_dataset = MockEnhancedDataset(1000, ['Tg', 'FFV', 'Tc', 'Density', 'Rg'])
    test_dataset = MockEnhancedDataset(100, [])  # No target columns like test data
    
    # Test with the problematic batch size
    batch_size = 64
    
    print(f"\n🔍 Testing training dataset with batch_size={batch_size}...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    
    try:
        for i, batch_data in enumerate(train_loader):
            batch_size_actual = batch_data.batch.max().item() + 1
            
            print(f"   Batch {i}: x.shape={batch_data.x.shape}, y.shape={batch_data.y.shape}, actual_batch_size={batch_size_actual}")
            
            if batch_data.y.shape[0] != batch_size_actual:
                print(f"   ❌ Shape mismatch in batch {i}: y.shape[0]={batch_data.y.shape[0]}, batch_size={batch_size_actual}")
                return False
            
            if i >= 2:  # Test a few batches
                break
                
        print("   ✅ Training dataset batches work correctly")
        
    except Exception as e:
        print(f"   ❌ Error in training dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🔍 Testing test dataset with batch_size={batch_size}...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    
    try:
        for i, batch_data in enumerate(test_loader):
            batch_size_actual = batch_data.batch.max().item() + 1
            
            print(f"   Batch {i}: x.shape={batch_data.x.shape}, y.shape={batch_data.y.shape}, actual_batch_size={batch_size_actual}")
            
            if batch_data.y.shape[0] != batch_size_actual:
                print(f"   ❌ Shape mismatch in batch {i}: y.shape[0]={batch_data.y.shape[0]}, batch_size={batch_size_actual}")
                return False
            
            if i >= 2:  # Test a few batches
                break
                
        print("   ✅ Test dataset batches work correctly")
        
    except Exception as e:
        print(f"   ❌ Error in test dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if test_realistic_collate():
        print("\n✅ Realistic collate test passed!")
        print("The collate function should work with the enhanced dataset.")
    else:
        print("\n❌ Realistic collate test failed!")
        print("There are still issues with the collate function.")