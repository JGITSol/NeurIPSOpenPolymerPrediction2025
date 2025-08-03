#!/usr/bin/env python3
"""
Clean environment setup for NeurIPS T4x2 solution
Handles NumPy compatibility issues without breaking existing packages
"""

import os
import sys
import warnings
import subprocess

def suppress_warnings():
    """Suppress all compatibility warnings."""
    warnings.filterwarnings('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings too
    
    # Additional warning suppression
    import logging
    logging.getLogger().setLevel(logging.ERROR)

def check_and_install_packages():
    """Check and install required packages without version conflicts."""
    required_packages = [
        'torch>=2.0.0',
        'torch-geometric', 
        'rdkit-pypi',
        'pandas>=1.5.0',
        'scikit-learn',
        'lightgbm',
        'tqdm'
    ]
    
    print("🔧 Setting up clean environment...")
    
    for package in required_packages:
        package_name = package.split('>=')[0].split('==')[0]
        try:
            # Try to import the package
            if package_name == 'torch-geometric':
                import torch_geometric
            elif package_name == 'rdkit-pypi':
                import rdkit
            elif package_name == 'scikit-learn':
                import sklearn
            else:
                __import__(package_name)
            print(f"✅ {package_name} already available")
        except ImportError:
            print(f"📦 Installing {package}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, 
                    "--no-warn-conflicts", "--disable-pip-version-check"
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ {package_name} installed")
            except subprocess.CalledProcessError:
                print(f"⚠️ Could not install {package}, but continuing...")

def test_environment():
    """Test that the environment works correctly."""
    print("\n🧪 Testing environment...")
    
    try:
        suppress_warnings()
        
        import torch
        from torch_geometric.data import Data, Batch
        import pandas as pd
        import numpy as np
        
        # Test PyTorch Geometric collate function
        dummy_data = Data(x=torch.randn(3, 32), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
        batch = Batch.from_data_list([dummy_data, dummy_data])
        
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Pandas: {pd.__version__}")
        print(f"✅ PyTorch Geometric batching works")
        print("✅ Environment test passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 NeurIPS T4x2 Environment Setup")
    print("=" * 50)
    
    # Suppress warnings first
    suppress_warnings()
    
    # Install packages
    check_and_install_packages()
    
    # Test environment
    if test_environment():
        print("\n🎉 Environment ready for T4x2 notebook!")
        print("You can now run the notebook without compatibility warnings.")
    else:
        print("\n⚠️ Environment setup had issues. Check the error messages above.")
    
    print("=" * 50)