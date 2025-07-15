"""Tests for the data validation module."""

import pandas as pd
import pytest
import numpy as np

from polymer_prediction.data.validation import (
    DataValidationConfig,
    validate_smiles,
    detect_outliers,
    validate_dataframe,
    clean_dataframe,
)


def test_validate_smiles():
    """Test SMILES validation."""
    # Valid SMILES
    assert validate_smiles("CC") is True
    assert validate_smiles("CCO") is True
    assert validate_smiles("c1ccccc1") is True
    
    # Invalid SMILES
    assert validate_smiles("invalid") is False
    assert validate_smiles("") is False
    assert validate_smiles("C(C") is False


def test_detect_outliers_iqr():
    """Test outlier detection using IQR method."""
    # Data with clear outliers
    data = np.array([1, 2, 3, 4, 5, 100, 200])
    outliers = detect_outliers(data, method="iqr", threshold=1.5)
    
    assert len(outliers) > 0
    assert 5 in outliers  # 100 should be detected as outlier
    assert 6 in outliers  # 200 should be detected as outlier


def test_detect_outliers_zscore():
    """Test outlier detection using Z-score method."""
    # Data with clear outliers
    data = np.array([1, 2, 3, 4, 5, 100])
    outliers = detect_outliers(data, method="zscore", threshold=2.0)
    
    assert len(outliers) > 0
    assert 5 in outliers  # 100 should be detected as outlier


@pytest.fixture
def valid_dataframe():
    """Create a valid DataFrame for testing."""
    return pd.DataFrame({
        'smiles': ['CC', 'CCO', 'c1ccccc1', 'CN(C)C'],
        'target_property': [1.0, 2.0, 3.0, 4.0]
    })


@pytest.fixture
def invalid_dataframe():
    """Create an invalid DataFrame for testing."""
    return pd.DataFrame({
        'smiles': ['CC', 'invalid_smiles', 'c1ccccc1', None],
        'target_property': [1.0, 2.0, 100.0, None]  # 100.0 is an outlier
    })


def test_validate_dataframe_valid(valid_dataframe):
    """Test validation of a valid DataFrame."""
    config = DataValidationConfig()
    report = validate_dataframe(valid_dataframe, config)
    
    assert report.is_valid is True
    assert report.total_samples == 4
    assert report.valid_samples == 4
    assert len(report.errors) == 0


def test_validate_dataframe_invalid(invalid_dataframe):
    """Test validation of an invalid DataFrame."""
    config = DataValidationConfig()
    report = validate_dataframe(invalid_dataframe, config)
    
    assert report.is_valid is False or len(report.warnings) > 0
    assert report.total_samples == 4
    assert len(report.invalid_smiles) > 0
    assert len(report.outliers) > 0


def test_validate_dataframe_empty():
    """Test validation of an empty DataFrame."""
    empty_df = pd.DataFrame()
    config = DataValidationConfig()
    report = validate_dataframe(empty_df, config)
    
    assert report.is_valid is False
    assert report.total_samples == 0
    assert "DataFrame is empty" in report.errors


def test_clean_dataframe(invalid_dataframe):
    """Test DataFrame cleaning."""
    config = DataValidationConfig()
    cleaned_df = clean_dataframe(invalid_dataframe, config, remove_invalid=True)
    
    # Should have fewer samples after cleaning
    assert len(cleaned_df) < len(invalid_dataframe)
    
    # Should not have any null values in required columns
    assert not cleaned_df['smiles'].isnull().any()
    assert not cleaned_df['target_property'].isnull().any()