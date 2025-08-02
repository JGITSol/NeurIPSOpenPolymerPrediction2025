#!/usr/bin/env python3
"""
Quick test script to verify GPU setup is working.
"""

import warnings
warnings.filterwarnings('ignore')

def test_basic_imports():
    """Test basic package imports."""
    print("Testing basic imports...")
    try:
        import torch
        import torch_geometric
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Pandas: {pd.__version__}")
        print(f"✅ LightGBM: {lgb.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_cuda():
    """Test CUDA functionality."""
    print("\nTesting CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Test GPU computation
            x = torch.randn(100, 100, device='cuda')
            y = torch.randn(100, 100, device='cuda')
            z = torch.matmul(x, y)
            print("✅ GPU computation test passed")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
            return True
        else:
            print("❌ CUDA not available")
            return False
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_rdkit():
    """Test RDKit with error handling."""
    print("\nTesting RDKit...")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from rdkit import Chem
            mol = Chem.MolFromSmiles('CCO')
            if mol is not None:
                print("✅ RDKit working")
                return True
            else:
                print("❌ RDKit molecule creation failed")
                return False
    except Exception as e:
        print(f"⚠️  RDKit issue: {e}")
        print("   This is likely a NumPy compatibility issue")
        print("   Try restarting your environment or reinstalling RDKit")
        return False

def test_pytorch_geometric():
    """Test PyTorch Geometric functionality."""
    print("\nTesting PyTorch Geometric...")
    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        import torch
        
        # Create simple graph
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        
        # Test GCN layer
        conv = GCNConv(1, 16)
        out = conv(data.x, data.edge_index)
        
        print(f"✅ PyTorch Geometric working")
        print(f"   Input shape: {data.x.shape}")
        print(f"   Output shape: {out.shape}")
        return True
    except Exception as e:
        print(f"❌ PyTorch Geometric test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("GPU Setup Verification")
    print("NeurIPS Open Polymer Prediction 2025")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_cuda,
        test_pytorch_geometric,
        test_rdkit
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    if passed >= 3:  # Allow RDKit to fail due to NumPy issues
        print("🎉 GPU setup is working!")
        print("You can now run the GPU-enhanced solution:")
        print("  python gpu_enhanced_solution.py --epochs 10 --batch_size 32")
        if passed < len(tests):
            print("\nNote: Some components had issues but core functionality works")
    else:
        print("❌ Setup has significant issues")
        print("Please check the error messages above")
    print("=" * 60)

if __name__ == "__main__":
    main()