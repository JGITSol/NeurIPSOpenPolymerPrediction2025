#!/usr/bin/env python3
"""
CRITICAL PERFORMANCE FIX: Remove spam output and fix CPU bottleneck
"""

import json
import re

def fix_performance_issues():
    """Fix spam output and CPU bottleneck issues."""
    
    # Read the notebook
    with open('neurips-t4x2-complete-solution-fixed.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Fix 1: Remove spam output from weighted_mae_loss
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'def weighted_mae_loss(predictions, targets, masks):' in source:
                print("🔧 Removing spam output from weighted_mae_loss...")
                
                # Remove the spam print statement
                new_source = source.replace(
                    'print(f"🔧 DataParallel fix: Adjusted predictions from {original_pred_size} to {actual_batch_size}")',
                    '# DataParallel fix applied silently'
                )
                
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print("✅ Removed spam output from weighted_mae_loss")
                break
    
    # Fix 2: Remove debug output from training loop
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if '🔍 First batch debug:' in source:
                print("🔧 Removing debug output from training loop...")
                
                # Remove all debug print statements
                lines = source.split('\n')
                new_lines = []
                skip_debug = False
                
                for line in lines:
                    if '# Debug: Check batch structure before loss calculation' in line:
                        skip_debug = True
                        continue
                    elif skip_debug and ('print(f"' in line or 'batch_size_actual' in line):
                        continue
                    elif skip_debug and 'loss = weighted_mae_loss' in line:
                        skip_debug = False
                        new_lines.append(line)
                    else:
                        new_lines.append(line)
                
                cell['source'] = [line + '\n' for line in new_lines[:-1]] + [new_lines[-1]]
                print("✅ Removed debug output from training loop")
                break
    
    # Fix 3: Optimize DataLoader for GPU performance
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'DataLoader(' in source and 'num_workers=0' in source:
                print("🔧 Optimizing DataLoader for GPU performance...")
                
                # Increase num_workers and add pin_memory for GPU efficiency
                new_source = source.replace(
                    'num_workers=0',
                    'num_workers=2, pin_memory=True, persistent_workers=True'
                )
                
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print("✅ Optimized DataLoader settings")
                break
    
    # Fix 4: Add GPU utilization optimization
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'BATCH_SIZE = 32  # Per GPU' in source:
                print("🔧 Adding GPU utilization optimizations...")
                
                # Add GPU optimization settings
                new_source = source.replace(
                    'print("🚀 T4 x2 GPU Configuration Loaded")',
                    '''# GPU Performance Optimizations
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # Clear GPU cache
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

print("🚀 T4 x2 GPU Configuration Loaded")'''
                )
                
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print("✅ Added GPU utilization optimizations")
                break
    
    # Save the fixed notebook
    with open('neurips-t4x2-complete-solution-fixed.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ Performance fixes applied successfully!")
    print("🎯 Training should now be GPU-bound with minimal console spam!")

if __name__ == "__main__":
    fix_performance_issues()