#!/usr/bin/env python3
"""
Debug script to identify the exact cause of the tensor shape mismatch
"""

import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import numpy as np

def create_debug_dataset(size=100):
    """Create a dataset similar to the enhanced dataset."""
    data_list = []
    
    for i in range(size):
        # Create random graph data
        num_nodes = np.random.randint(3, 8)
        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        
        data = Data(x=x, edge_index=edge_index)
        
        # Always add y and mask (like the enhanced dataset should)
        targets = torch.randn(5)
        masks = torch.ones(5)
        data.y = targets
        data.mask = masks
        
        data_list.append(data)
    
    return data_list

def debug_collate_batch(batch):
    """Debug version of collate function with extensive logging."""
    print(f"\n🔍 Collate function called with batch size: {len(batch)}")
    
    # Filter out None items and ensure all items are Data objects
    valid_batch = []
    for i, item in enumerate(batch):
        if item is not None and hasattr(item, 'x') and hasattr(item, 'edge_index'):
            print(f"   Item {i}: x.shape={item.x.shape}, has_y={hasattr(item, 'y')}, has_mask={hasattr(item, 'mask')}")
            if hasattr(item, 'y'):
                print(f"            y.shape={item.y.shape}")
            if hasattr(item, 'mask'):
                print(f"            mask.shape={item.mask.shape}")
            
            # Ensure all items have y and mask attributes
            if not hasattr(item, 'y') or item.y is None:
                item.y = torch.zeros(5, dtype=torch.float)
                print(f"            Added zero y: {item.y.shape}")
            if not hasattr(item, 'mask') or item.mask is None:
                item.mask = torch.zeros(5, dtype=torch.float)
                print(f"            Added zero mask: {item.mask.shape}")
            
            valid_batch.append(item)
    
    print(f"   Valid batch size: {len(valid_batch)}")
    
    # If no valid items, create a dummy batch
    if len(valid_batch) == 0:
        dummy_data = Data(x=torch.zeros((1, 32)), edge_index=torch.empty((2, 0), dtype=torch.long))
        dummy_data.y = torch.zeros(5, dtype=torch.float)
        dummy_data.mask = torch.zeros(5, dtype=torch.float)
        valid_batch = [dummy_data]
        print("   Created dummy batch")
    
    # Create batch using PyTorch Geometric's Batch.from_data_list
    try:
        # Store y and mask before creating batch to prevent PyG from concatenating them
        y_list = [item.y.clone() for item in valid_batch]
        mask_list = [item.mask.clone() for item in valid_batch]
        
        print(f"   Collected y_list: {len(y_list)} items, shapes: {[y.shape for y in y_list[:3]]}")
        print(f"   Collected mask_list: {len(mask_list)} items, shapes: {[m.shape for m in mask_list[:3]]}")
        
        # Temporarily remove y and mask from items to prevent PyG concatenation
        for item in valid_batch:
            if hasattr(item, 'y'):
                delattr(item, 'y')
            if hasattr(item, 'mask'):
                delattr(item, 'mask')
        
        # Create the batch for graph structure only
        batch_data = Batch.from_data_list(valid_batch)
        print(f"   Created batch_data: batch.shape={batch_data.batch.shape}")
        
        # Add y and mask as graph-level attributes manually
        if y_list:
            batch_data.y = torch.stack(y_list, dim=0)  # Shape: (batch_size, num_properties)
            print(f"   Stacked y: {batch_data.y.shape}")
        if mask_list:
            batch_data.mask = torch.stack(mask_list, dim=0)  # Shape: (batch_size, num_properties)
            print(f"   Stacked mask: {batch_data.mask.shape}")
        
        # Debug: Check final shapes
        batch_size = batch_data.batch.max().item() + 1
        print(f"   Final batch_size: {batch_size}")
        if hasattr(batch_data, 'y'):
            print(f"   Final y.shape: {batch_data.y.shape}")
        if hasattr(batch_data, 'mask'):
            print(f"   Final mask.shape: {batch_data.mask.shape}")
        
        if hasattr(batch_data, 'y') and batch_data.y.shape[0] != batch_size:
            print(f"   ❌ SHAPE MISMATCH: batch_size={batch_size}, y.shape[0]={batch_data.y.shape[0]}")
        else:
            print(f"   ✅ Shapes match correctly")
        
        return batch_data
        
    except Exception as e:
        print(f'   ❌ Error in collate_batch: {e}')
        import traceback
        traceback.print_exc()
        # Return a dummy batch as fallback
        dummy_data = Data(x=torch.zeros((1, 32)), edge_index=torch.empty((2, 0), dtype=torch.long))
        dummy_data.y = torch.zeros((1, 5))
        dummy_data.mask = torch.zeros((1, 5))
        return dummy_data

def test_debug_collate():
    """Test the debug collate function."""
    print("🧪 Testing debug collate function...")
    
    # Create test dataset
    dataset = create_debug_dataset(20)
    
    # Test with batch size 64 (the problematic size)
    batch_size = 64
    print(f"\n🔍 Testing with batch_size={batch_size}...")
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=debug_collate_batch)
    
    # Test one batch
    try:
        for batch_data in loader:
            print(f"\n📊 Final batch results:")
            print(f"   x shape: {batch_data.x.shape}")
            print(f"   batch shape: {batch_data.batch.shape}")
            print(f"   y shape: {batch_data.y.shape}")
            print(f"   mask shape: {batch_data.mask.shape}")
            
            # Check if shapes would work with model
            batch_size_actual = batch_data.batch.max().item() + 1
            print(f"   Actual batch size: {batch_size_actual}")
            
            if batch_data.y.shape[0] != batch_size_actual:
                print(f"   ❌ This would cause the tensor shape mismatch!")
                return False
            else:
                print(f"   ✅ Shapes are correct for model training")
            
            break  # Only test first batch
            
    except Exception as e:
        print(f"   ❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    if test_debug_collate():
        print("\n✅ Debug test passed - collate function should work correctly")
    else:
        print("\n❌ Debug test failed - there are still issues")