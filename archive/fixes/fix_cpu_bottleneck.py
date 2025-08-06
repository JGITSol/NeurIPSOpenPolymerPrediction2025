#!/usr/bin/env python3
"""
CRITICAL CPU BOTTLENECK FIX: Optimize data processing for GPU training
"""

import json

def fix_cpu_bottleneck():
    """Fix CPU bottleneck by optimizing data processing."""
    
    # Read the notebook
    with open('neurips-t4x2-complete-solution-fixed.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Fix 1: Optimize collate function for GPU efficiency
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'def collate_batch(batch):' in source:
                print("🔧 Optimizing collate function for GPU efficiency...")
                
                # Replace with optimized collate function
                optimized_collate = '''def collate_batch(batch):
    """Optimized collate function for GPU training."""
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Use PyTorch Geometric's built-in batching (much faster)
    try:
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)
    except Exception as e:
        print(f"Batch collation error: {e}")
        return None
'''
                
                # Find and replace the collate function
                lines = source.split('\n')
                new_lines = []
                in_collate_function = False
                function_indent = 0
                
                for line in lines:
                    if 'def collate_batch(batch):' in line:
                        in_collate_function = True
                        function_indent = len(line) - len(line.lstrip())
                        # Add the optimized function
                        new_lines.extend(optimized_collate.split('\n'))
                        continue
                    
                    if in_collate_function:
                        # Check if we're still in the function
                        if line.strip() == '':
                            continue  # Skip empty lines
                        elif line.startswith(' ' * (function_indent + 1)) or line.strip().startswith('#') or line.strip().startswith('"""'):
                            continue  # Skip function content
                        else:
                            # We've reached the end of the function
                            in_collate_function = False
                            new_lines.append(line)
                    else:
                        new_lines.append(line)
                
                cell['source'] = [line + '\n' for line in new_lines[:-1]] + [new_lines[-1]]
                print("✅ Optimized collate function")
                break
    
    # Fix 2: Optimize batch size for better GPU utilization
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'BATCH_SIZE = 32  # Per GPU' in source:
                print("🔧 Optimizing batch size for GPU utilization...")
                
                # Increase batch size for better GPU utilization
                new_source = source.replace(
                    'BATCH_SIZE = 32  # Per GPU',
                    'BATCH_SIZE = 48  # Per GPU - optimized for T4 memory'
                )
                
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print("✅ Optimized batch size")
                break
    
    # Fix 3: Add prefetch_factor for DataLoader
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            
            if 'persistent_workers=True' in source:
                print("🔧 Adding prefetch optimization...")
                
                # Add prefetch_factor for better pipeline
                new_source = source.replace(
                    'persistent_workers=True',
                    'persistent_workers=True, prefetch_factor=4'
                )
                
                cell['source'] = new_source.split('\n')
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                print("✅ Added prefetch optimization")
                break
    
    # Save the fixed notebook
    with open('neurips-t4x2-complete-solution-fixed.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print("✅ CPU bottleneck fixes applied successfully!")
    print("🚀 Training should now be much more GPU-bound!")

if __name__ == "__main__":
    fix_cpu_bottleneck()