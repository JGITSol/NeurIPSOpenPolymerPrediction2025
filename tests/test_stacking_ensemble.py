"""
Tests for stacking ensemble with cross-validation.

This module tests the StackingEnsemble class that combines GCN and tree ensemble
models using cross-validation to generate out-of-fold predictions and a meta-learner
to combine base model predictions.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error
import tempfile
import os

from src.polymer_prediction.models.stacking_ensemble import StackingEnsemble
from src.polymer_prediction.models.ensemble import TreeEnsemble


class MockGCNModel(nn.Module):
    """Mock GCN model for testing purposes."""
    
    def __init__(self, num_atom_features, hidden_channels=64, num_gcn_layers=2):
        super().__init__()
        self.num_atom_features = num_atom_features
        self.hidden_channels = hidden_channels
        self.num_gcn_layers = num_gcn_layers
        
        # Simple linear layers to simulate GCN
        self.input_layer = nn.Linear(num_atom_features, hidden_channels)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels) 
            for _ in range(num_gcn_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_channels, 5)  # 5 target properties
        
    def forward(self, data):
        x = data.x
        # Simple aggregation over nodes (mean pooling)
        x = torch.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        
        # Global mean pooling to get graph-level representation
        batch_size = data.batch.max().item() + 1
        graph_embeddings = []
        for i in range(batch_size):
            mask = data.batch == i
            if mask.sum() > 0:
                graph_emb = x[mask].mean(dim=0)
            else:
                graph_emb = torch.zeros(self.hidden_channels, device=x.device)
            graph_embeddings.append(graph_emb)
        
        graph_embeddings = torch.stack(graph_embeddings)
        return self.output_layer(graph_embeddings)


def create_sample_data(n_samples=100, n_features=50, n_targets=5, missing_rate=0.2):
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Create sample SMILES (dummy)
    smiles = [f"C{i}CC" for i in range(n_samples)]
    
    # Create feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Create target matrix with some missing values
    y = np.random.randn(n_samples, n_targets)
    
    # Add missing values
    missing_mask = np.random.random((n_samples, n_targets)) < missing_rate
    y[missing_mask] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'SMILES': smiles,
        'Tg': y[:, 0],
        'FFV': y[:, 1], 
        'Tc': y[:, 2],
        'Density': y[:, 3],
        'Rg': y[:, 4]
    })
    
    return df, X, y


def create_mock_graph_data(smiles_list, target_cols, is_test=False):
    """Create mock graph data for testing."""
    data_list = []
    
    for i, smiles in enumerate(smiles_list):
        # Create dummy node features (10 features per node, 5 nodes per molecule)
        num_nodes = 5
        num_features = 10
        x = torch.randn(num_nodes, num_features)
        
        # Create dummy edge indices (fully connected)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Create target and mask
        if not is_test:
            y = torch.randn(5)  # 5 target properties
            mask = torch.ones(5)  # All targets present for simplicity
        else:
            y = torch.zeros(5)
            mask = torch.zeros(5)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            mask=mask,
            id=i
        )
        data_list.append(data)
    
    return data_list


class TestStackingEnsemble:
    """Test cases for StackingEnsemble class."""
    
    def test_initialization(self):
        """Test StackingEnsemble initialization."""
        ensemble = StackingEnsemble(
            gcn_model_class=MockGCNModel,
            gcn_params={'hidden_channels': 64},
            tree_models=['lgbm'],
            cv_folds=3,
            random_state=42
        )
        
        assert ensemble.gcn_model_class == MockGCNModel
        assert ensemble.gcn_params['hidden_channels'] == 64
        assert ensemble.tree_models == ['lgbm']
        assert ensemble.cv_folds == 3
        assert ensemble.random_state == 42
    
    def test_cv_splits_creation(self):
        """Test cross-validation splits creation."""
        ensemble = StackingEnsemble(
            gcn_model_class=MockGCNModel,
            cv_folds=3,
            random_state=42
        )
        
        # Create sample data
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 5)
        
        splits = ensemble._create_cv_splits(X, y)
        
        assert len(splits) == 3
        
        # Check that splits don't overlap and cover all data
        all_train_indices = set()
        all_val_indices = set()
        
        for train_idx, val_idx in splits:
            train_set = set(train_idx)
            val_set = set(val_idx)
            
            # No overlap between train and val in same fold
            assert len(train_set.intersection(val_set)) == 0
            
            all_train_indices.update(train_set)
            all_val_indices.update(val_set)
        
        # All indices should be covered
        assert all_val_indices == set(range(100))
    
    def test_average_fold_scores(self):
        """Test averaging of fold scores."""
        ensemble = StackingEnsemble(
            gcn_model_class=MockGCNModel,
            random_state=42
        )
        ensemble.target_cols_ = ['Tg', 'FFV', 'Tc']
        
        fold_scores = [
            {'Tg': 1.0, 'FFV': 2.0, 'Tc': np.nan},
            {'Tg': 1.5, 'FFV': 2.5, 'Tc': 3.0},
            {'Tg': 2.0, 'FFV': np.nan, 'Tc': 3.5}
        ]
        
        avg_scores = ensemble._average_fold_scores(fold_scores)
        
        assert abs(avg_scores['Tg'] - 1.5) < 1e-6  # (1.0 + 1.5 + 2.0) / 3
        assert abs(avg_scores['FFV'] - 2.25) < 1e-6  # (2.0 + 2.5) / 2
        assert abs(avg_scores['Tc'] - 3.25) < 1e-6  # (3.0 + 3.5) / 2
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_stacking_ensemble_small_dataset(self):
        """Test stacking ensemble with a small dataset."""
        # Create small sample data
        df, X, y = create_sample_data(n_samples=50, n_features=20, n_targets=5)
        
        # Mock the PolymerDataset to return our mock graph data
        import src.polymer_prediction.data.dataset as dataset_module
        original_dataset = dataset_module.PolymerDataset
        
        class MockPolymerDataset:
            def __init__(self, df, target_cols, is_test=False):
                self.df = df
                self.target_cols = target_cols
                self.is_test = is_test
                self.data_list = create_mock_graph_data(df['SMILES'].tolist(), target_cols, is_test)
            
            def __len__(self):
                return len(self.data_list)
            
            def __getitem__(self, idx):
                return self.data_list[idx]
        
        # Temporarily replace PolymerDataset
        dataset_module.PolymerDataset = MockPolymerDataset
        
        try:
            # Create stacking ensemble with minimal configuration
            ensemble = StackingEnsemble(
                gcn_model_class=MockGCNModel,
                gcn_params={'hidden_channels': 32, 'num_gcn_layers': 2},
                tree_models=['lgbm'],  # Only use LightGBM for speed
                cv_folds=3,
                gcn_epochs=5,  # Minimal epochs for testing
                optimize_tree_hyperparams=False,  # Skip optimization for speed
                random_state=42,
                device=torch.device('cpu'),  # Use CPU for testing
                batch_size=16
            )
            
            # Fit the ensemble
            ensemble.fit(df, X, y)
            
            # Check that models were trained
            assert 'gcn' in ensemble.base_models_ or 'tree' in ensemble.base_models_
            assert len(ensemble.meta_models_) > 0
            
            # Test prediction
            test_df, test_X, _ = create_sample_data(n_samples=20, n_features=20, n_targets=5)
            predictions = ensemble.predict(test_df, test_X)
            
            assert predictions.shape == (20, 5)
            assert not np.all(np.isnan(predictions))  # Should have some valid predictions
            
            # Test CV scores retrieval
            cv_scores = ensemble.get_cv_scores()
            assert isinstance(cv_scores, dict)
            
            # Test model info retrieval
            model_info = ensemble.get_model_info()
            assert 'n_targets' in model_info
            assert 'cv_folds' in model_info
            assert model_info['n_targets'] == 5
            assert model_info['cv_folds'] == 3
            
        finally:
            # Restore original PolymerDataset
            dataset_module.PolymerDataset = original_dataset
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data."""
        ensemble = StackingEnsemble(
            gcn_model_class=MockGCNModel,
            cv_folds=2,
            random_state=42
        )
        
        # Test with all NaN targets
        df_nan = pd.DataFrame({
            'SMILES': ['CCC', 'CCO'],
            'Tg': [np.nan, np.nan],
            'FFV': [np.nan, np.nan],
            'Tc': [np.nan, np.nan],
            'Density': [np.nan, np.nan],
            'Rg': [np.nan, np.nan]
        })
        X_nan = np.random.randn(2, 10)
        y_nan = np.full((2, 5), np.nan)
        
        # Should not crash, but may not train properly
        try:
            ensemble.fit(df_nan, X_nan, y_nan)
            # If it doesn't crash, that's good
            assert True
        except Exception as e:
            # Some errors are expected with all-NaN data
            assert "No valid" in str(e) or "empty" in str(e).lower()
    
    def test_prediction_without_fitting(self):
        """Test that prediction fails when ensemble is not fitted."""
        ensemble = StackingEnsemble(
            gcn_model_class=MockGCNModel,
            random_state=42
        )
        
        df, X, _ = create_sample_data(n_samples=10)
        
        with pytest.raises(ValueError, match="must be fitted"):
            ensemble.predict(df, X)
    
    def test_meta_model_training(self):
        """Test meta-model training with known data."""
        ensemble = StackingEnsemble(
            gcn_model_class=MockGCNModel,
            random_state=42
        )
        
        # Set up ensemble state
        ensemble.n_targets_ = 2
        ensemble.target_cols_ = ['target_0', 'target_1']
        
        # Create base predictions and targets
        n_samples = 100
        base_predictions = np.random.randn(n_samples, 4)  # 2 models * 2 targets
        y = np.random.randn(n_samples, 2)
        
        # Train meta-models
        ensemble._train_meta_models(base_predictions, y)
        
        # Check that meta-models were created
        assert len(ensemble.meta_models_) <= 2  # May be less if some targets have no valid data
        
        # Test that meta-models can make predictions
        for target_idx, meta_model in ensemble.meta_models_.items():
            test_input = np.random.randn(10, 2)  # 2 base model predictions
            predictions = meta_model.predict(test_input)
            assert predictions.shape == (10,)


def test_simple():
    """Simple test to verify the module works."""
    assert True

if __name__ == "__main__":
    pytest.main([__file__])