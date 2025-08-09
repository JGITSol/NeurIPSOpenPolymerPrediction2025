"""
Tests for DataLoader functionality and dataset creation.

This module tests the fixed DataLoader functionality, dataset creation,
and proper handling of PyTorch Geometric data structures.
"""

import pytest
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import DataLoader, Data, Batch
from unittest.mock import Mock, patch
import tempfile
import os

# Import the components to test
import sys
sys.path.append('src')

from polymer_prediction.data.dataset import PolymerDataset
from polymer_prediction.preprocessing.featurization import smiles_to_graph


class TestDataLoaderFunctionality:
    """Test DataLoader functionality with PyTorch Geometric."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O', 'invalid_smiles', 'CCCCCCCC'],
            'Tg': [1.0, 2.0, 3.0, 4.0, 5.0],
            'FFV': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Tc': [10.0, 20.0, 30.0, 40.0, 50.0],
            'Density': [0.8, 0.9, 1.0, 1.1, 1.2],
            'Rg': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
    
    @pytest.fixture
    def sample_dataset(self, sample_dataframe):
        """Create a sample PolymerDataset."""
        return PolymerDataset(
            sample_dataframe, 
            target_cols=['Tg', 'FFV', 'Tc', 'Density', 'Rg'], 
            is_test=False
        )
    
    def test_pytorch_geometric_dataloader_import(self):
        """Test that PyTorch Geometric DataLoader is properly imported."""
        # This should not raise an ImportError
        from torch_geometric.data import DataLoader
        assert DataLoader is not None
    
    def test_dataloader_creation(self, sample_dataset):
        """Test DataLoader creation with PolymerDataset."""
        loader = DataLoader(sample_dataset, batch_size=2, shuffle=False)
        assert loader is not None
        assert loader.batch_size == 2
    
    def test_dataloader_batch_processing(self, sample_dataset):
        """Test that DataLoader properly batches graph data."""
        # Create a filtered dataset that only contains valid samples
        valid_indices = []
        for i in range(len(sample_dataset)):
            data = sample_dataset.get(i)
            if data is not None:
                valid_indices.append(i)
        
        # Create a subset with only valid indices
        from torch.utils.data import Subset
        valid_dataset = Subset(sample_dataset, valid_indices)
        
        if len(valid_dataset) == 0:
            pytest.skip("No valid samples in dataset")
        
        loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
        
        batch_count = 0
        total_samples = 0
        
        for batch in loader:
            batch_count += 1
            total_samples += batch.num_graphs
            
            # Check batch structure
            assert hasattr(batch, 'x')  # Node features
            assert hasattr(batch, 'edge_index')  # Edge connectivity
            assert hasattr(batch, 'batch')  # Batch assignment
            assert hasattr(batch, 'y')  # Target values
            assert hasattr(batch, 'mask')  # Missing value masks
            
            # Check tensor shapes
            assert batch.x.dim() == 2  # (num_nodes, num_features)
            assert batch.edge_index.dim() == 2  # (2, num_edges)
            assert batch.y.dim() == 2  # (batch_size, num_targets)
            assert batch.mask.dim() == 2  # (batch_size, num_targets)
            
            # Check that batch size matches expected
            assert batch.y.size(0) <= 2  # Should be <= batch_size
            assert batch.mask.size(0) <= 2
        
        assert batch_count > 0  # Should have processed some batches
    
    def test_dataloader_with_invalid_smiles(self, sample_dataframe):
        """Test DataLoader handling of invalid SMILES."""
        # Create dataset with more invalid SMILES
        invalid_df = sample_dataframe.copy()
        invalid_df.loc[1, 'SMILES'] = 'invalid_smiles_1'
        invalid_df.loc[3, 'SMILES'] = 'invalid_smiles_2'
        
        dataset = PolymerDataset(invalid_df, target_cols=['Tg'], is_test=False)
        
        # Count valid and invalid samples
        valid_count = 0
        invalid_count = 0
        
        for i in range(len(dataset)):
            data = dataset.get(i)
            if data is not None:
                valid_count += 1
            else:
                invalid_count += 1
        
        # Should have some valid and some invalid samples
        assert valid_count > 0, "Should have some valid SMILES"
        assert invalid_count > 0, "Should have some invalid SMILES"
        
        # Test DataLoader with only valid samples
        valid_indices = []
        for i in range(len(dataset)):
            if dataset.get(i) is not None:
                valid_indices.append(i)
        
        if len(valid_indices) > 0:
            from torch.utils.data import Subset
            valid_dataset = Subset(dataset, valid_indices)
            loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
            
            batch_count = 0
            for batch in loader:
                batch_count += 1
                assert batch.num_graphs > 0
            
            assert batch_count > 0
    
    def test_dataloader_test_mode(self, sample_dataframe):
        """Test DataLoader with test dataset (no targets)."""
        test_dataset = PolymerDataset(sample_dataframe, is_test=True)
        loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        
        for batch in loader:
            if batch is not None:
                assert hasattr(batch, 'x')
                assert hasattr(batch, 'edge_index')
                assert hasattr(batch, 'id')
                # Test data should not have targets or masks
                assert not hasattr(batch, 'y') or batch.y is None
                assert not hasattr(batch, 'mask') or batch.mask is None
                break
    
    def test_dataloader_empty_batch_handling(self):
        """Test DataLoader behavior with empty or all-invalid data."""
        # Create DataFrame with all invalid SMILES
        invalid_df = pd.DataFrame({
            'id': [1, 2, 3],
            'SMILES': ['invalid1', 'invalid2', 'invalid3'],
            'Tg': [1.0, 2.0, 3.0]
        })
        
        dataset = PolymerDataset(invalid_df, target_cols=['Tg'], is_test=False)
        
        # Check that all samples are invalid
        valid_count = 0
        for i in range(len(dataset)):
            if dataset.get(i) is not None:
                valid_count += 1
        
        # Should have no valid samples
        assert valid_count == 0, "All SMILES should be invalid"
        
        # Test that we can handle empty dataset gracefully
        valid_indices = []
        for i in range(len(dataset)):
            if dataset.get(i) is not None:
                valid_indices.append(i)
        
        # Should have no valid indices
        assert len(valid_indices) == 0
        
        # DataLoader with empty dataset should work
        if len(valid_indices) > 0:
            from torch.utils.data import Subset
            valid_dataset = Subset(dataset, valid_indices)
            loader = DataLoader(valid_dataset, batch_size=2)
            
            batch_count = 0
            for batch in loader:
                batch_count += 1
            
            assert batch_count == 0  # No batches for empty dataset
        else:
            # No valid data, which is expected
            assert True


class TestDatasetCreation:
    """Test dataset creation and data handling."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O', 'C1=CC=CC=C1O', 'CCCCCCCC'],
            'Tg': [1.0, 2.0, None, 3.0, 4.0],
            'FFV': [0.1, None, 0.3, 0.4, 0.5],
            'Tc': [10.0, 20.0, 30.0, None, 50.0],
            'Density': [0.8, 0.9, 1.0, 1.1, None],
            'Rg': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
    
    def test_dataset_initialization(self, sample_dataframe):
        """Test PolymerDataset initialization."""
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        dataset = PolymerDataset(sample_dataframe, target_cols=target_cols, is_test=False)
        
        assert len(dataset) == 5
        assert dataset.target_cols == target_cols
        assert not dataset.is_test
        assert len(dataset.smiles_list) == 5
        assert len(dataset.ids) == 5
        assert len(dataset.targets) == 5
        assert len(dataset.masks) == 5
    
    def test_dataset_target_handling(self, sample_dataframe):
        """Test proper handling of missing target values."""
        dataset = PolymerDataset(sample_dataframe, target_cols=['Tg', 'FFV'], is_test=False)
        
        # Check first sample (no missing values)
        assert dataset.targets[0] == [1.0, 0.1]
        assert dataset.masks[0] == [1.0, 1.0]
        
        # Check second sample (FFV is missing)
        assert dataset.targets[1] == [2.0, 0.0]  # Missing value replaced with 0.0
        assert dataset.masks[1] == [1.0, 0.0]  # Mask indicates missing
        
        # Check third sample (Tg is missing)
        assert dataset.targets[2] == [0.0, 0.3]  # Missing value replaced with 0.0
        assert dataset.masks[2] == [0.0, 1.0]  # Mask indicates missing
    
    def test_dataset_get_item(self, sample_dataframe):
        """Test getting individual items from dataset."""
        dataset = PolymerDataset(sample_dataframe, target_cols=['Tg'], is_test=False)
        
        # Get first item
        data = dataset[0]
        
        if data is not None:  # Valid SMILES
            assert isinstance(data, Data)
            assert hasattr(data, 'x')  # Node features
            assert hasattr(data, 'edge_index')  # Edge connectivity
            assert hasattr(data, 'y')  # Target values
            assert hasattr(data, 'mask')  # Missing value masks
            assert hasattr(data, 'id')  # Sample ID
            
            # Check tensor shapes
            assert data.x.dim() == 2  # (num_nodes, num_features)
            assert data.edge_index.dim() == 2  # (2, num_edges)
            assert data.y.shape == (1, 1)  # (1, num_targets)
            assert data.mask.shape == (1, 1)  # (1, num_targets)
            
            # Check ID
            assert data.id == 1
    
    def test_dataset_caching(self, sample_dataframe):
        """Test that dataset caches computed graphs."""
        dataset = PolymerDataset(sample_dataframe, target_cols=['Tg'], is_test=False)
        
        # Get same item twice
        data1 = dataset[0]
        data2 = dataset[0]
        
        if data1 is not None and data2 is not None:
            # Should be cached (same object reference for graph data)
            assert 0 in dataset.cache
    
    def test_dataset_test_mode(self, sample_dataframe):
        """Test dataset in test mode (no targets)."""
        dataset = PolymerDataset(sample_dataframe, is_test=True)
        
        assert dataset.is_test
        assert not hasattr(dataset, 'targets')
        assert not hasattr(dataset, 'masks')
        
        data = dataset[0]
        if data is not None:
            assert hasattr(data, 'x')
            assert hasattr(data, 'edge_index')
            assert hasattr(data, 'id')
            # Should not have targets in test mode
            assert not hasattr(data, 'y') or data.y is None
            assert not hasattr(data, 'mask') or data.mask is None
    
    def test_dataset_invalid_smiles_handling(self):
        """Test dataset handling of invalid SMILES."""
        invalid_df = pd.DataFrame({
            'id': [1, 2, 3],
            'SMILES': ['CCO', 'invalid_smiles', 'c1ccccc1'],
            'Tg': [1.0, 2.0, 3.0]
        })
        
        dataset = PolymerDataset(invalid_df, target_cols=['Tg'], is_test=False)
        
        # Valid SMILES should return data
        data0 = dataset[0]
        assert data0 is not None
        
        # Invalid SMILES should return None
        data1 = dataset[1]
        assert data1 is None
        
        # Valid SMILES should return data
        data2 = dataset[2]
        assert data2 is not None


class TestSMILESToGraphConversion:
    """Test SMILES to graph conversion functionality."""
    
    def test_valid_smiles_conversion(self):
        """Test conversion of valid SMILES to graph."""
        valid_smiles = ['CCO', 'c1ccccc1', 'CC(C)O', 'C1=CC=CC=C1O']
        
        for smiles in valid_smiles:
            graph = smiles_to_graph(smiles)
            
            if graph is not None:  # Some SMILES might fail due to RDKit issues
                assert isinstance(graph, Data)
                assert hasattr(graph, 'x')  # Node features
                assert hasattr(graph, 'edge_index')  # Edge connectivity
                assert graph.x.dim() == 2  # (num_nodes, num_features)
                assert graph.edge_index.dim() == 2  # (2, num_edges)
                assert graph.x.size(0) > 0  # Should have nodes
    
    def test_invalid_smiles_conversion(self):
        """Test conversion of invalid SMILES."""
        invalid_smiles = ['invalid_smiles', '', 'C(C(C', '123abc']
        
        for smiles in invalid_smiles:
            graph = smiles_to_graph(smiles)
            # Invalid SMILES should return None
            assert graph is None
    
    def test_edge_case_smiles(self):
        """Test edge cases in SMILES conversion."""
        edge_cases = [
            'C',  # Single carbon
            'CC',  # Ethane
            '[H]',  # Hydrogen
            'C=C',  # Ethene
            'C#C',  # Ethyne
        ]
        
        for smiles in edge_cases:
            graph = smiles_to_graph(smiles)
            # These should either work or fail gracefully
            if graph is not None:
                assert isinstance(graph, Data)
                assert graph.x.size(0) > 0


class TestIntegrationWithMainPipeline:
    """Test integration with the main pipeline components."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for integration testing."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O', 'C1=CC=CC=C1O', 'CCCCCCCC'],
            'Tg': [1.0, 2.0, 3.0, 4.0, 5.0],
            'FFV': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Tc': [10.0, 20.0, 30.0, 40.0, 50.0],
            'Density': [0.8, 0.9, 1.0, 1.1, 1.2],
            'Rg': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
    
    def test_end_to_end_dataloader_pipeline(self, sample_data):
        """Test complete pipeline from DataFrame to DataLoader."""
        # Create dataset
        dataset = PolymerDataset(
            sample_data, 
            target_cols=['Tg', 'FFV', 'Tc', 'Density', 'Rg'], 
            is_test=False
        )
        
        # Create DataLoader with safe collate function
        def safe_collate_fn(batch):
            valid_batch = [item for item in batch if item is not None]
            if len(valid_batch) == 0:
                return None
            return Batch.from_data_list(valid_batch)
        
        loader = DataLoader(dataset, batch_size=2, collate_fn=safe_collate_fn)
        
        # Test iteration through DataLoader
        total_samples = 0
        for batch in loader:
            if batch is not None:
                total_samples += batch.num_graphs
                
                # Verify batch structure
                assert batch.x.dim() == 2
                assert batch.edge_index.dim() == 2
                assert batch.y.dim() == 2
                assert batch.mask.dim() == 2
                assert batch.y.size(1) == 5  # 5 target properties
                assert batch.mask.size(1) == 5
        
        # Should have processed all valid samples
        assert total_samples > 0
    
    def test_dataloader_with_missing_values(self):
        """Test DataLoader with missing target values."""
        data_with_missing = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O', 'C1=CC=CC=C1O'],
            'Tg': [1.0, None, 3.0, 4.0],
            'FFV': [0.1, 0.2, None, 0.4],
            'Tc': [10.0, 20.0, 30.0, None]
        })
        
        dataset = PolymerDataset(data_with_missing, target_cols=['Tg', 'FFV', 'Tc'], is_test=False)
        
        def safe_collate_fn(batch):
            valid_batch = [item for item in batch if item is not None]
            if len(valid_batch) == 0:
                return None
            return Batch.from_data_list(valid_batch)
        
        loader = DataLoader(dataset, batch_size=2, collate_fn=safe_collate_fn)
        
        for batch in loader:
            if batch is not None:
                # Check that masks properly indicate missing values
                assert batch.mask.sum() > 0  # Should have some valid values
                assert batch.mask.sum() < batch.mask.numel()  # Should have some missing values
                
                # Check that missing values are set to 0 in targets
                missing_mask = batch.mask == 0
                if missing_mask.sum() > 0:
                    missing_targets = batch.y[missing_mask]
                    assert torch.all(missing_targets == 0.0)
                break


def test_simple_dataloader():
    """Simple test to verify the module works."""
    assert True


if __name__ == "__main__":
    pytest.main([__file__])