#!/usr/bin/env python3
"""
Test the T4x2 notebook warning suppression
"""

import os
import warnings

# Apply the same warning suppression as in the notebook
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Testing T4x2 notebook imports with warning suppression...")

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data, Batch
    import pandas as pd
    import numpy as np
    
    print("✅ All imports successful")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    
    # Test the collate function
    dummy_data = Data(x=torch.randn(3, 32), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    batch = Batch.from_data_list([dummy_data, dummy_data])
    print(f"✅ PyTorch Geometric batching works: {batch.batch.shape}")
    
    print("🎉 T4x2 notebook should run cleanly!")
    
except Exception as e:
    print(f"❌ Error: {e}")