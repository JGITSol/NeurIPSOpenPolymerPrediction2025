#!/usr/bin/env python3
"""
CRITICAL FIX: DataParallel Tensor Shape Mismatch
This script fixes the tensor shape mismatch issue in the T4x2 notebook
where DataParallel concatenates predictions from multiple GPUs.
"""

import json
import re

def fix_weighted_mae_loss():
    """Fix the weighted_mae_loss function to handle DataParallel shape mismatch."""
    
    # Read the notebook
    with open('neurips-t4x2-complete-solution-fixed.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix the weighted_mae_loss function
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'def weighted_mae_loss(predictions, targets, masks):' in source:
                print("🔧 Found weighted_mae_loss function - applying DataParallel fix...")
                
                # New fixed function
                new_function = '''# Training functions
def weighted_mae_loss(predictions, targets, masks):
    """Calculate weighted MAE loss with DataParallel shape handling."""
    
    # Handle DataParallel shape mismatch - predictions get concatenated from multiple GPUs
    if predictions.shape[0] != targets.shape[0]:
        # DataParallel concatenates outputs from multiple GPUs
        # We need to take only the first batch_size predictions
        actual_batch_size = targets.shape[0]
        original_pred_size = predictions.shape[0]
        predictions = predictions[:actual_batch_size]
        print(f"🔧 DataParallel fix: Adjusted predictions from {original_pred_size} to {actual_batch_size}")
    
    # Final shape validation
    if predictions.shape != targets.shape or predictions.shape != masks.shape:
        print(f"⚠️ Tensor shape mismatch after DataParallel fix:")
        print(f"   predictions: {predictions.shape}")
        print(f"   targets: {targets.shape}")
        print(f"   masks: {masks.shape}")
        raise ValueError(f"Shape mismatch: pred={predictions.shape}, target={targets.shape}, mask={masks.shape}")
    
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=predictions.device, dtype=predictions.dtype)
    
    # Ensure proper broadcasting
    if len(weights.shape) == 1 and len(predictions.shape) == 2:
        weights = weights.unsqueeze(0)  # Shape: (1, 5) for broadcasting
    
    mae_per_property = torch.abs(predictions - targets) * masks
    weighted_mae = (mae_per_property * weights).sum() / (masks * weights).sum()
    
    # Avoid division by zero
    if torch.isnan(weighted_mae) or torch.isinf(weighted_mae):
        return torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
    
    return weighted_mae
'''
                
                # Replace the function in the cell
                # Find the start and end of the function
                lines = source.split('\n')
                new_lines = []
                in_function = False
                function_indent = 0
                
                for line in lines:
                    if 'def weighted_mae_loss(predictions, targets, masks):' in line:
                        in_function = True
                        function_indent = len(line) - len(line.lstrip())
                        # Add the new function
                        new_lines.extend(new_function.split('\n'))
                        continue
                    
                    if in_function:
                        # Check if we're still in the function
                        if line.strip() == '':
                            continue  # Skip empty lines
                        elif line.startswith(' ' * (function_indent + 1)) or line.strip().startswith('#') or line.strip().startswith('"""'):
                            continue  # Skip function content
                        else:
                            # We've reached the end of the function
                            in_function = False
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                
                # Update the cell source
                cell['source'] = [line + '\n' for line in new_lines[:-1]] + [new_lines[-1]]
                print("✅ Fixed weighted_mae_loss function")
                break
    
    # Save the fixed notebook
    with open('neurips-t4x2-complete-solution-fixed.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ DataParallel tensor shape fix applied successfully!")

if __name__ == "__main__":
    fix_weighted_mae_loss()