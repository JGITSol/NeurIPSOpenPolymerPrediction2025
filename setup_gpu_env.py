#!/usr/bin/env python3
"""
Setup script for GPU-enhanced environment.
Installs optimized packages for CUDA 11.8 and ≤6 GB VRAM.
"""

import subprocess
import sys
import os
import platform


def run_command(cmd, check=True):
    """Run a command and handle errors."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def install_gpu_packages():
    """Install GPU-optimized packages."""
    print("Installing GPU-optimized packages...")
    
    # PyTorch with CUDA 11.8
    torch_cmd = (
        "pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    
    if not run_command(torch_cmd):
        print("❌ Failed to install PyTorch with CUDA")
        return False
    
    # PyTorch Geometric with CUDA support
    pyg_cmd = (
        "pip install torch-scatter==2.1.2+cu118 torch-sparse==0.6.18+cu118 "
        "torch-cluster==1.6.3+cu118 torch-spline-conv==1.2.2+cu118 "
        "--index-url https://data.pyg.org/whl/torch-2.2.2+cu118.html"
    )
    
    if not run_command(pyg_cmd):
        print("❌ Failed to install PyTorch Geometric extensions")
        return False
    
    # Install PyTorch Geometric
    if not run_command("pip install torch-geometric==2.5.3"):
        print("❌ Failed to install PyTorch Geometric")
        return False
    
    # Install other requirements
    if not run_command("pip install -r requirements-gpu.txt"):
        print("❌ Failed to install other requirements")
        return False
    
    print("✅ All packages installed successfully!")
    return True


def verify_installation():
    """Verify the installation."""
    print("\nVerifying installation...")
    
    try:
        import torch
        import torch_geometric
        import rdkit
        import lightgbm
        import pandas as pd
        import numpy as np
        
        print("✅ All core packages imported successfully")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            
            # Test GPU memory
            try:
                x = torch.randn(1000, 1000, device='cuda')
                y = torch.randn(1000, 1000, device='cuda')
                z = torch.matmul(x, y)
                print("✅ GPU computation test passed")
                del x, y, z
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"❌ GPU computation test failed: {e}")
                return False
        else:
            print("⚠️  CUDA not available, will use CPU")
        
        # Test PyTorch Geometric
        try:
            from torch_geometric.data import Data
            from torch_geometric.nn import GCNConv
            print("✅ PyTorch Geometric working")
        except Exception as e:
            print(f"❌ PyTorch Geometric test failed: {e}")
            return False
        
        # Test RDKit
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles('CCO')
            if mol is not None:
                print("✅ RDKit working")
            else:
                print("❌ RDKit test failed")
                return False
        except Exception as e:
            print(f"❌ RDKit test failed: {e}")
            return False
        
        print("✅ Installation verification complete!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("GPU-Enhanced Environment Setup")
    print("NeurIPS Open Polymer Prediction 2025")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in a virtual environment (recommended to use one)")
    
    # Install packages
    if not install_gpu_packages():
        print("❌ Installation failed!")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Verification failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Setup complete!")
    print("You can now run the GPU-enhanced solution:")
    print("  python gpu_enhanced_solution.py --epochs 50")
    print("=" * 60)


if __name__ == "__main__":
    main()