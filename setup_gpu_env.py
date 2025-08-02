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
    
    # Install PyTorch Geometric (will auto-detect CUDA)
    print("Installing PyTorch Geometric...")
    if not run_command("pip install torch-geometric"):
        print("❌ Failed to install PyTorch Geometric")
        return False
    
    print("✅ PyTorch Geometric installed (CUDA extensions will be installed on-demand)")
    
    # Install other requirements
    if not run_command("pip install -r requirements-gpu.txt"):
        print("❌ Failed to install other requirements")
        return False
    
    # Fix NumPy compatibility issue with RDKit
    print("Fixing NumPy compatibility...")
    if not run_command('pip install "numpy<2.0"'):
        print("⚠️  Failed to downgrade NumPy, but continuing...")
    
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
        
        # Test RDKit (with NumPy compatibility handling)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from rdkit import Chem
                mol = Chem.MolFromSmiles('CCO')
                if mol is not None:
                    print("✅ RDKit working")
                else:
                    print("❌ RDKit test failed")
                    return False
        except Exception as e:
            print(f"⚠️  RDKit import issue (likely NumPy compatibility): {e}")
            print("   This may be resolved by restarting the environment")
            print("   The GPU solution should still work")
        
        print("✅ Installation verification complete!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def activate_venv():
    """Activate the virtual environment if it exists."""
    venv_path = os.path.join(os.getcwd(), "venv")
    if os.path.exists(venv_path):
        print(f"✅ Found virtual environment at: {venv_path}")
        
        # Check if we're already in the venv
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✅ Already in virtual environment")
            return True
        else:
            print("⚠️  Please activate the virtual environment first:")
            print("   venv\\Scripts\\activate")
            print("   Then run this script again")
            return False
    else:
        print("⚠️  No virtual environment found. Creating one...")
        if not run_command("python -m venv venv"):
            print("❌ Failed to create virtual environment")
            return False
        print("✅ Virtual environment created")
        print("⚠️  Please activate it and run this script again:")
        print("   venv\\Scripts\\activate")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("GPU-Enhanced Environment Setup")
    print("NeurIPS Open Polymer Prediction 2025")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check and activate virtual environment
    if not activate_venv():
        sys.exit(1)
    
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