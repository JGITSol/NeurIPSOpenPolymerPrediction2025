#!/usr/bin/env python3
"""
Test script to verify wmae_loss function works correctly with different tensor shapes
"""

import torch
import numpy as np

def wmae_loss(pred, target, mask, weights=None):
    """Weighted Mean Absolute Error loss."""
    if weights is None:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    # Ensure weights tensor has the right shape for broadcasting
    weights = torch.tensor(weights, device=pred.device, dtype=pred.dtype)
    if len(weights.shape) == 1 and len(pred.shape) == 2:
        weights = weights.unsqueeze(0)  # Shape: (1, 5) for broadcasting with (batch_size, 5)
    
    diff = torch.abs(pred - target) * mask
    weighted_diff = diff * weights
    
    # Calculate weighted sum and normalization
    numerator = torch.sum(weighted_diff)
    denominator = torch.sum(mask * weights)
    
    # Avoid division by zero
    if denominator == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    
    return numerator / denominator

def test_wmae_loss():
    """Test wmae_loss with different batch sizes."""
    print("🧪 Testing wmae_loss function...")
    
    # Test cases with different batch sizes
    test_cases = [
        {"batch_size": 1, "name": "Single sample"},
        {"batch_size": 16, "name": "Small batch"},
        {"batch_size": 32, "name": "Medium batch"},
        {"batch_size": 64, "name": "Large batch"},
    ]
    
    for case in test_cases:
        batch_size = case["batch_size"]
        name = case["name"]
        
        print(f"\n🔍 Testing {name} (batch_size={batch_size})...")
        
        # Create test tensors
        pred = torch.randn(batch_size, 5)
        target = torch.randn(batch_size, 5)
        mask = torch.ones(batch_size, 5)  # All valid
        
        print(f"   pred shape: {pred.shape}")
        print(f"   target shape: {target.shape}")
        print(f"   mask shape: {mask.shape}")
        
        try:
            loss = wmae_loss(pred, target, mask)
            print(f"   ✅ Loss calculated: {loss.item():.4f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    # Test with partial masks
    print(f"\n🔍 Testing with partial masks...")
    batch_size = 32
    pred = torch.randn(batch_size, 5)
    target = torch.randn(batch_size, 5)
    mask = torch.rand(batch_size, 5) > 0.5  # Random mask
    mask = mask.float()
    
    try:
        loss = wmae_loss(pred, target, mask)
        print(f"   ✅ Partial mask loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ Partial mask error: {e}")
        return False
    
    # Test with all zeros mask (edge case)
    print(f"\n🔍 Testing with zero mask (edge case)...")
    mask_zero = torch.zeros(batch_size, 5)
    
    try:
        loss = wmae_loss(pred, target, mask_zero)
        print(f"   ✅ Zero mask loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ Zero mask error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if test_wmae_loss():
        print("\n✅ All wmae_loss tests passed!")
    else:
        print("\n❌ Some wmae_loss tests failed!")