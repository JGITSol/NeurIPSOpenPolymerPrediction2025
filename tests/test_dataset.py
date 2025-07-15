"""Tests for the dataset module."""

import pandas as pd
import pytest
import torch

from polymer_prediction.data.dataset import PolymerDataset


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'smiles': ['CC', 'CCO', 'c1ccccc1', 'invalid_smiles'],
        'target_property': [1.0, 2.0, 3.0, 4.0]
    })


def test_polymer_dataset_creation(sample_dataframe):
    """Test PolymerDataset creation."""
    dataset = PolymerDataset(sample_dataframe, target_col='target_property')
    
    assert len(dataset) == 4
    assert dataset.smiles_list == ['CC', 'CCO', 'c1ccccc1', 'invalid_smiles']
    assert dataset.targets == [1.0, 2.0, 3.0, 4.0]


def test_polymer_dataset_get_valid_sample(sample_dataframe):
    """Test getting a valid sample from the dataset."""
    dataset = PolymerDataset(sample_dataframe, target_col='target_property')
    
    # Get first sample (CC)
    data = dataset.get(0)
    
    assert data is not None
    assert hasattr(data, 'x')
    assert hasattr(data, 'edge_index')
    assert hasattr(data, 'y')
    assert data.y.item() == 1.0


def test_polymer_dataset_get_invalid_sample(sample_dataframe):
    """Test getting an invalid sample from the dataset."""
    dataset = PolymerDataset(sample_dataframe, target_col='target_property')
    
    # Get last sample (invalid_smiles)
    data = dataset.get(3)
    
    assert data is None


def test_polymer_dataset_caching(sample_dataframe):
    """Test that the dataset caches computed graphs."""
    dataset = PolymerDataset(sample_dataframe, target_col='target_property')
    
    # Get sample twice
    data1 = dataset.get(0)
    data2 = dataset.get(0)
    
    # Should be the same object due to caching
    assert data1 is data2
    assert 0 in dataset.cache