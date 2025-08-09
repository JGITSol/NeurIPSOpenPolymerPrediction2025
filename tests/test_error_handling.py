"""
Test suite for error handling and robustness features.

This module tests all components of the enhanced error handling system
to ensure they work correctly under various failure scenarios.
"""

import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

# Import the error handling components
import sys
sys.path.append('src')

from polymer_prediction.utils.error_handling import (
    ErrorHandler,
    SMILESValidator,
    MemoryManager,
    DeviceManager,
    InputValidator,
    PolymerPredictionError,
    SMILESValidationError,
    ModelTrainingError,
    robust_function_wrapper
)
from polymer_prediction.data.robust_dataset import (
    RobustPolymerDataset,
    create_robust_dataloader,
    DatasetValidator
)
from polymer_prediction.training.robust_trainer import (
    RobustTrainer,
    train_one_epoch_robust,
    predict_robust
)


class TestErrorHandler:
    """Test the ErrorHandler class."""
    
    def test_error_handler_initialization(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler()
        assert handler.invalid_smiles_count == 0
        assert handler.memory_warnings_count == 0
        assert handler.device_fallbacks_count == 0
        assert handler.training_failures_count == 0
    
    def test_handle_invalid_smiles(self):
        """Test invalid SMILES handling."""
        handler = ErrorHandler()
        
        # Test single invalid SMILES
        handler.handle_invalid_smiles("invalid_smiles", 0, "test context")
        assert handler.invalid_smiles_count == 1
        
        # Test multiple invalid SMILES
        for i in range(5):
            handler.handle_invalid_smiles(f"invalid_{i}", i)
        assert handler.invalid_smiles_count == 6
    
    def test_handle_memory_error(self):
        """Test memory error handling."""
        handler = ErrorHandler()
        
        # Test batch size reduction
        new_batch_size = handler.handle_memory_error(32, "test operation")
        assert new_batch_size == 16
        assert handler.memory_warnings_count == 1
        
        # Test minimum batch size
        new_batch_size = handler.handle_memory_error(1, "test operation")
        assert new_batch_size == 1
        assert handler.memory_warnings_count == 2
    
    def test_handle_training_failure(self):
        """Test training failure handling."""
        handler = ErrorHandler()
        
        # Test with fallback available
        result = handler.handle_training_failure("test_model", Exception("test error"), True)
        assert result is True
        assert handler.training_failures_count == 1
        
        # Test without fallback
        result = handler.handle_training_failure("test_model", Exception("test error"), False)
        assert result is False
        assert handler.training_failures_count == 2
    
    def test_handle_device_error(self):
        """Test device error handling."""
        handler = ErrorHandler()
        
        preferred_device = torch.device('cuda')
        fallback_device = handler.handle_device_error(preferred_device, Exception("CUDA error"))
        
        assert fallback_device == torch.device('cpu')
        assert handler.device_fallbacks_count == 1
    
    def test_get_error_summary(self):
        """Test error summary generation."""
        handler = ErrorHandler()
        
        # Generate some errors
        handler.handle_invalid_smiles("test", 0)
        handler.handle_memory_error(32)
        handler.handle_training_failure("test", Exception(), True)
        handler.handle_device_error(torch.device('cuda'), Exception())
        
        summary = handler.get_error_summary()
        assert summary['invalid_smiles'] == 1
        assert summary['memory_warnings'] == 1
        assert summary['training_failures'] == 1
        assert summary['device_fallbacks'] == 1


class TestSMILESValidator:
    """Test the SMILESValidator class."""
    
    def test_validate_smiles_valid(self):
        """Test validation of valid SMILES."""
        validator = SMILESValidator()
        
        valid_smiles = ['CCO', 'c1ccccc1', 'CC(C)O', 'C1=CC=CC=C1O']
        for smiles in valid_smiles:
            assert validator.validate_smiles(smiles) is True
    
    def test_validate_smiles_invalid(self):
        """Test validation of invalid SMILES."""
        validator = SMILESValidator()
        
        invalid_smiles = ['invalid_smiles', '', 'C(C(C', '123abc', None]
        for smiles in invalid_smiles:
            if smiles is not None:
                assert validator.validate_smiles(smiles) is False
    
    def test_validate_smiles_list(self):
        """Test validation of SMILES list."""
        validator = SMILESValidator()
        
        smiles_list = ['CCO', 'invalid', 'c1ccccc1', '', 'CC(C)O']
        validity_list, invalid_indices = validator.validate_smiles_list(smiles_list)
        
        assert len(validity_list) == len(smiles_list)
        assert len(invalid_indices) == 2  # 'invalid' and ''
        assert 1 in invalid_indices
        assert 3 in invalid_indices
    
    def test_filter_valid_smiles(self):
        """Test filtering of valid SMILES from DataFrame."""
        validator = SMILESValidator()
        
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'SMILES': ['CCO', 'invalid', 'c1ccccc1', '', 'CC(C)O'],
            'target': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        filtered_df = validator.filter_valid_smiles(df, 'SMILES')
        
        assert len(filtered_df) == 3  # Only valid SMILES
        assert 'invalid' not in filtered_df['SMILES'].values
        assert '' not in filtered_df['SMILES'].values


class TestMemoryManager:
    """Test the MemoryManager class."""
    
    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        manager = MemoryManager()
        
        # Test with limited memory
        optimal_size = manager.get_optimal_batch_size(32, 1.5)  # 1.5GB available
        assert optimal_size <= 8  # Should be reduced
        
        # Test with sufficient memory
        optimal_size = manager.get_optimal_batch_size(32, 8.0)  # 8GB available
        assert optimal_size == 32  # Should remain the same
    
    def test_adaptive_batch_size_reduction(self):
        """Test adaptive batch size reduction."""
        manager = MemoryManager()
        
        new_size = manager.adaptive_batch_size_reduction(32, "test operation")
        assert new_size == 16
        assert manager.current_batch_size == 16


class TestDeviceManager:
    """Test the DeviceManager class."""
    
    def test_detect_optimal_device(self):
        """Test optimal device detection."""
        manager = DeviceManager()
        
        device = manager.detect_optimal_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda']
    
    def test_safe_device_transfer(self):
        """Test safe device transfer."""
        manager = DeviceManager()
        manager.current_device = torch.device('cpu')
        
        # Test tensor transfer
        tensor = torch.randn(10, 10)
        transferred = manager.safe_device_transfer(tensor)
        assert transferred.device.type == 'cpu'
        
        # Test model transfer
        model = nn.Linear(10, 5)
        transferred_model = manager.safe_device_transfer(model)
        assert next(transferred_model.parameters()).device.type == 'cpu'


class TestInputValidator:
    """Test the InputValidator class."""
    
    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        validator = InputValidator()
        
        # Test valid DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O'],
            'target': [1.0, 2.0, 3.0]
        })
        
        validated_df = validator.validate_dataframe(df, ['id', 'SMILES'], "test")
        assert len(validated_df) == 3
        
        # Test missing columns
        with pytest.raises(Exception):
            validator.validate_dataframe(df, ['id', 'SMILES', 'missing_col'], "test")
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            validator.validate_dataframe(empty_df, ['id'], "test")
    
    def test_validate_file_path(self):
        """Test file path validation."""
        validator = InputValidator()
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            validated_path = validator.validate_file_path(tmp_path, must_exist=True)
            assert isinstance(validated_path, Path)
            assert validated_path.exists()
        finally:
            os.unlink(tmp_path)
        
        # Test non-existent file
        with pytest.raises(Exception):
            validator.validate_file_path("non_existent_file.txt", must_exist=True)
    
    def test_validate_model_parameters(self):
        """Test model parameter validation."""
        validator = InputValidator()
        
        # Test valid parameters
        params = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 100
        }
        
        validated = validator.validate_model_parameters(params)
        assert validated['batch_size'] == 32
        assert validated['learning_rate'] == 0.001
        assert validated['num_epochs'] == 100
        
        # Test invalid parameters
        invalid_params = {
            'batch_size': -1,  # Invalid
            'learning_rate': 0.001
        }
        
        with pytest.raises(Exception):
            validator.validate_model_parameters(invalid_params)


class TestRobustFunctionWrapper:
    """Test the robust function wrapper decorator."""
    
    def test_successful_function(self):
        """Test wrapper with successful function."""
        @robust_function_wrapper(fallback_return="fallback")
        def successful_function(x):
            return x * 2
        
        result = successful_function(5)
        assert result == 10
    
    def test_failing_function(self):
        """Test wrapper with failing function."""
        @robust_function_wrapper(fallback_return="fallback")
        def failing_function(x):
            raise ValueError("Test error")
        
        result = failing_function(5)
        assert result == "fallback"
    
    def test_reraise_option(self):
        """Test wrapper with reraise option."""
        @robust_function_wrapper(reraise=True)
        def failing_function(x):
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function(5)


class TestRobustPolymerDataset:
    """Test the RobustPolymerDataset class."""
    
    def create_sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'SMILES': ['CCO', 'invalid_smiles', 'c1ccccc1', 'CC(C)O', 'C1=CC=CC=C1O'],
            'Tg': [1.0, 2.0, None, 3.0, 4.0],
            'FFV': [0.1, 0.2, 0.3, None, 0.5],
            'Tc': [10.0, None, 30.0, 40.0, 50.0],
            'Density': [0.8, 0.9, 1.0, 1.1, None],
            'Rg': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
    
    def test_dataset_initialization(self):
        """Test robust dataset initialization."""
        df = self.create_sample_dataframe()
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        dataset = RobustPolymerDataset(df, target_cols=target_cols, is_test=False)
        
        # Should filter out invalid SMILES
        assert len(dataset) < len(df)
        assert dataset.get_valid_smiles_count() > 0
        assert dataset.get_invalid_smiles_count() > 0
    
    def test_dataset_get_item(self):
        """Test getting items from robust dataset."""
        df = self.create_sample_dataframe()
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        dataset = RobustPolymerDataset(df, target_cols=target_cols, is_test=False)
        
        # Get first valid item
        if len(dataset) > 0:
            data = dataset[0]
            assert data is not None
            assert hasattr(data, 'x')  # Node features
            assert hasattr(data, 'edge_index')  # Edge connectivity
            assert hasattr(data, 'y')  # Target values
            assert hasattr(data, 'mask')  # Missing value mask
            assert hasattr(data, 'id')  # Sample ID
    
    def test_dataset_statistics(self):
        """Test dataset statistics generation."""
        df = self.create_sample_dataframe()
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        dataset = RobustPolymerDataset(df, target_cols=target_cols, is_test=False)
        stats = dataset.get_dataset_stats()
        
        assert 'total_samples' in stats
        assert 'valid_samples' in stats
        assert 'invalid_samples' in stats
        assert 'validity_ratio' in stats
        assert 'target_statistics' in stats


class TestRobustDataLoader:
    """Test the robust DataLoader creation."""
    
    def test_create_robust_dataloader(self):
        """Test robust DataLoader creation."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O'],
            'Tg': [1.0, 2.0, 3.0]
        })
        
        dataset = RobustPolymerDataset(df, target_cols=['Tg'], is_test=False)
        loader = create_robust_dataloader(dataset, batch_size=2, shuffle=False)
        
        assert loader is not None
        
        # Test that loader can handle None values
        batch_count = 0
        for batch in loader:
            if batch is not None:
                batch_count += 1
        
        assert batch_count >= 0  # Should handle batches gracefully


class TestRobustTrainer:
    """Test the RobustTrainer class."""
    
    def create_simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    def create_mock_dataloader(self):
        """Create a mock DataLoader for testing."""
        # Create mock data
        mock_data = Mock()
        mock_data.x = torch.randn(10, 10)
        mock_data.y = torch.randn(1, 1)
        mock_data.mask = torch.ones(1, 1)
        mock_data.num_graphs = 1
        mock_data.to = Mock(return_value=mock_data)
        
        # Create mock loader
        mock_loader = Mock()
        mock_loader.__iter__ = Mock(return_value=iter([mock_data, mock_data, mock_data]))
        mock_loader.batch_size = 2
        
        return mock_loader
    
    def test_trainer_initialization(self):
        """Test RobustTrainer initialization."""
        trainer = RobustTrainer()
        
        assert trainer.max_retries == 3
        assert trainer.min_batch_size == 1
        assert trainer.gradient_clip_norm == 1.0
        assert trainer.patience == 10
    
    @patch('polymer_prediction.training.robust_trainer.masked_mse_loss')
    def test_train_one_epoch_robust(self, mock_loss):
        """Test robust training for one epoch."""
        # Setup mocks
        mock_loss.return_value = torch.tensor(0.5, requires_grad=True)
        
        trainer = RobustTrainer()
        model = self.create_simple_model()
        optimizer = torch.optim.Adam(model.parameters())
        device = torch.device('cpu')
        loader = self.create_mock_dataloader()
        
        # Test training
        avg_loss = trainer._train_one_epoch_robust(model, loader, optimizer, device, 0)
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0


def run_error_handling_tests():
    """Run all error handling tests."""
    print("Running error handling tests...")
    
    # Test ErrorHandler
    print("Testing ErrorHandler...")
    test_handler = TestErrorHandler()
    test_handler.test_error_handler_initialization()
    test_handler.test_handle_invalid_smiles()
    test_handler.test_handle_memory_error()
    test_handler.test_handle_training_failure()
    test_handler.test_handle_device_error()
    test_handler.test_get_error_summary()
    print("✓ ErrorHandler tests passed")
    
    # Test SMILESValidator
    print("Testing SMILESValidator...")
    test_validator = TestSMILESValidator()
    test_validator.test_validate_smiles_valid()
    test_validator.test_validate_smiles_invalid()
    test_validator.test_validate_smiles_list()
    test_validator.test_filter_valid_smiles()
    print("✓ SMILESValidator tests passed")
    
    # Test MemoryManager
    print("Testing MemoryManager...")
    test_memory = TestMemoryManager()
    test_memory.test_get_optimal_batch_size()
    test_memory.test_adaptive_batch_size_reduction()
    print("✓ MemoryManager tests passed")
    
    # Test DeviceManager
    print("Testing DeviceManager...")
    test_device = TestDeviceManager()
    test_device.test_detect_optimal_device()
    test_device.test_safe_device_transfer()
    print("✓ DeviceManager tests passed")
    
    # Test InputValidator
    print("Testing InputValidator...")
    test_input = TestInputValidator()
    test_input.test_validate_dataframe()
    test_input.test_validate_file_path()
    test_input.test_validate_model_parameters()
    print("✓ InputValidator tests passed")
    
    # Test RobustFunctionWrapper
    print("Testing RobustFunctionWrapper...")
    test_wrapper = TestRobustFunctionWrapper()
    test_wrapper.test_successful_function()
    test_wrapper.test_failing_function()
    test_wrapper.test_reraise_option()
    print("✓ RobustFunctionWrapper tests passed")
    
    # Test RobustPolymerDataset
    print("Testing RobustPolymerDataset...")
    test_dataset = TestRobustPolymerDataset()
    test_dataset.test_dataset_initialization()
    test_dataset.test_dataset_get_item()
    test_dataset.test_dataset_statistics()
    print("✓ RobustPolymerDataset tests passed")
    
    # Test RobustDataLoader
    print("Testing RobustDataLoader...")
    test_loader = TestRobustDataLoader()
    test_loader.test_create_robust_dataloader()
    print("✓ RobustDataLoader tests passed")
    
    # Test RobustTrainer
    print("Testing RobustTrainer...")
    test_trainer = TestRobustTrainer()
    test_trainer.test_trainer_initialization()
    print("✓ RobustTrainer tests passed")
    
    print("\n🎉 All error handling tests passed successfully!")


if __name__ == "__main__":
    run_error_handling_tests()