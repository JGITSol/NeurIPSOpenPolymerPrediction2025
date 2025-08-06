#!/usr/bin/env python3
"""
Fix for multi-GPU device placement error in DataParallel setup
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

def fix_multi_gpu_device_error():
    """
    Common solutions for 'Expected all tensors to be on the same device' error
    in multi-GPU DataParallel setups.
    """
    
    print("🔧 Multi-GPU Device Placement Error Fixes")
    print("=" * 50)
    
    print("\n1. 📍 Ensure Model is on Primary Device Before DataParallel:")
    print("""
    # WRONG:
    model = MyModel()
    model = DataParallel(model)
    model.to(device)  # Too late!
    
    # CORRECT:
    model = MyModel()
    model.to(device)  # Move to primary device first
    model = DataParallel(model)
    """)
    
    print("\n2. 📍 Ensure Input Data is on Primary Device:")
    print("""
    # In training loop:
    for batch in dataloader:
        # WRONG:
        batch = batch.to('cuda:1')  # Wrong device
        
        # CORRECT:
        batch = batch.to(device)  # Use primary device (cuda:0)
        # or
        batch = batch.to('cuda:0')  # Explicitly use primary device
    """)
    
    print("\n3. 📍 Fix Custom Forward Method:")
    print("""
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )
        
        def forward(self, batch):
            # WRONG: Assuming batch is on correct device
            x = batch.x
            x = self.atom_encoder(x)  # May fail if x is on wrong device
            
            # CORRECT: Ensure input is on same device as model
            x = batch.x.to(next(self.parameters()).device)
            x = self.atom_encoder(x)
    """)
    
    print("\n4. 📍 Fix DataLoader Device Placement:")
    print("""
    # In collate function or dataset:
    def collate_fn(batch):
        # Process batch...
        batch_data = create_batch(batch)
        
        # WRONG: Don't specify device in collate
        # batch_data = batch_data.to('cuda:1')
        
        # CORRECT: Let DataParallel handle device placement
        return batch_data  # Device placement handled by DataParallel
    """)
    
    print("\n5. 📍 Complete Fix Example:")
    print("""
    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Model setup
    model = MyModel()
    model.to(device)  # Move to primary device FIRST
    
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)  # Then wrap with DataParallel
    
    # Training loop
    for batch in dataloader:
        batch = batch.to(device)  # Move batch to primary device
        
        optimizer.zero_grad()
        predictions = model(batch)  # DataParallel handles distribution
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
    """)
    
    print("\n6. 📍 Debug Device Placement:")
    print("""
    # Add debugging to identify device mismatches:
    def debug_devices(batch, model):
        print(f"Batch device: {batch.x.device}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Check all model parameters are on same device
        devices = {p.device for p in model.parameters()}
        if len(devices) > 1:
            print(f"WARNING: Model parameters on multiple devices: {devices}")
    """)

if __name__ == "__main__":
    fix_multi_gpu_device_error()