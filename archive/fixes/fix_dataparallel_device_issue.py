#!/usr/bin/env python3
"""
Fix the DataParallel device placement issue in the kaggle-ready notebook.
"""

import json

def fix_dataparallel_device_issue():
    """Fix the device placement issue in DataParallel training."""
    
    # Read the notebook
    with open('neurips-t4x2-kaggle-ready.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Fix 1: Update model forward method to handle device placement properly
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'def forward(self, data):' in source and 'T4PolyGIN' in source:
                print("🔧 Fixing model forward method for proper device handling...")
                
                # Replace the forward method with proper device handling
                new_source = source.replace(
                    '''    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Get device (handle DataParallel StopIteration issue)
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = self.device  # Fallback for DataParallel replicas
        
        # Input projection
        x = self.input_proj(x)''',
                    '''    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Get device (handle DataParallel StopIteration issue)
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = self.device  # Fallback for DataParallel replicas
        
        # Ensure all tensors are on the same device as model parameters
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        
        # Input projection
        x = self.input_proj(x)'''
                )
                
                cell['source'] = new_source.split('\\n')
                cell['source'] = [line + '\\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print("✅ Fixed model forward method")
                break
    
    # Fix 2: Update training function to ensure proper device placement
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'def train_epoch(model, train_loader, optimizer, device):' in source:
                print("🔧 Fixing training function for proper device handling...")
                
                # Replace the batch.to(device) section with more robust device handling
                new_source = source.replace(
                    '''        # Move batch to primary device (cuda:0)
        batch = batch.to(device)
        optimizer.zero_grad()''',
                    '''        # Move batch to primary device with explicit device placement
        batch = batch.to(device, non_blocking=True)
        
        # Ensure all batch components are on the correct device
        if hasattr(batch, 'x'):
            batch.x = batch.x.to(device, non_blocking=True)
        if hasattr(batch, 'edge_index'):
            batch.edge_index = batch.edge_index.to(device, non_blocking=True)
        if hasattr(batch, 'batch'):
            batch.batch = batch.batch.to(device, non_blocking=True)
        if hasattr(batch, 'y'):
            batch.y = batch.y.to(device, non_blocking=True)
        if hasattr(batch, 'mask'):
            batch.mask = batch.mask.to(device, non_blocking=True)
        
        optimizer.zero_grad()'''
                )
                
                cell['source'] = new_source.split('\\n')
                cell['source'] = [line + '\\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print("✅ Fixed training function")
                break
    
    # Fix 3: Update evaluation function similarly
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'def evaluate_model(model, val_loader, device, scaler=None):' in source:
                print("🔧 Fixing evaluation function for proper device handling...")
                
                # Replace the batch.to(device) section
                new_source = source.replace(
                    '''            batch = batch.to(device)''',
                    '''            # Move batch to device with explicit component handling
            batch = batch.to(device, non_blocking=True)
            
            # Ensure all batch components are on the correct device
            if hasattr(batch, 'x'):
                batch.x = batch.x.to(device, non_blocking=True)
            if hasattr(batch, 'edge_index'):
                batch.edge_index = batch.edge_index.to(device, non_blocking=True)
            if hasattr(batch, 'batch'):
                batch.batch = batch.batch.to(device, non_blocking=True)
            if hasattr(batch, 'y'):
                batch.y = batch.y.to(device, non_blocking=True)
            if hasattr(batch, 'mask'):
                batch.mask = batch.mask.to(device, non_blocking=True)'''
                )
                
                cell['source'] = new_source.split('\\n')
                cell['source'] = [line + '\\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print("✅ Fixed evaluation function")
                break
    
    # Fix 4: Update model initialization to use only cuda:0 as primary device
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if '# Move to device' in source and 'model = model.to(device)' in source:
                print("🔧 Fixing model initialization for DataParallel...")
                
                # Replace model initialization with proper DataParallel setup
                new_source = source.replace(
                    '''# Move to device
model = model.to(device)

# Setup DataParallel for multi-GPU training
if torch.cuda.device_count() > 1:
    print(f"🚀 Enabling DataParallel for {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
    effective_batch_size = BATCH_SIZE * torch.cuda.device_count()
    print(f"   Effective batch size: {effective_batch_size}")
else:
    effective_batch_size = BATCH_SIZE
    print(f"   Single GPU training, batch size: {effective_batch_size}")''',
                    '''# Move model to primary device (cuda:0) FIRST
primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(primary_device)

# Setup DataParallel for multi-GPU training AFTER moving to primary device
if torch.cuda.device_count() > 1:
    print(f"🚀 Enabling DataParallel for {torch.cuda.device_count()} GPUs")
    print(f"   Primary device: {primary_device}")
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    effective_batch_size = BATCH_SIZE * torch.cuda.device_count()
    print(f"   Effective batch size: {effective_batch_size}")
    # Update device to primary device for data loading
    device = primary_device
else:
    effective_batch_size = BATCH_SIZE
    print(f"   Single GPU training, batch size: {effective_batch_size}")'''
                )
                
                cell['source'] = new_source.split('\\n')
                cell['source'] = [line + '\\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print("✅ Fixed model initialization")
                break
    
    # Fix 5: Update device setup to use cuda:0 as primary
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")' in source:
                print("🔧 Fixing device setup for DataParallel...")
                
                # Replace device setup
                new_source = source.replace(
                    'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
                    '# Use cuda:0 as primary device for DataParallel\\ndevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")'
                )
                
                cell['source'] = new_source.split('\\n')
                cell['source'] = [line + '\\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print("✅ Fixed device setup")
                break
    
    # Save the fixed notebook
    with open('neurips-t4x2-kaggle-ready.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ DataParallel device placement fixes applied successfully!")
    print("🎯 Key fixes:")
    print("   ✅ Model forward method: Explicit device placement for all tensors")
    print("   ✅ Training function: Robust batch device handling")
    print("   ✅ Evaluation function: Consistent device placement")
    print("   ✅ Model initialization: Primary device (cuda:0) setup")
    print("   ✅ Device configuration: cuda:0 as primary device")
    print("🚀 Should now work properly with T4 x2 DataParallel!")

if __name__ == "__main__":
    fix_dataparallel_device_issue()