"""
Tests for tree ensemble models.

This module tests the LightGBM, XGBoost, and CatBoost wrappers,
hyperparameter optimization, and ensemble training pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.polymer_prediction.models.ensemble import (
    LightGBMWrapper,
    XGBoostWrapper,
    CatBoostWrapper,
    HyperparameterOptimizer,
    TreeEnsemble,
    LIGHTGBM_AVAILABLE,
    XGBOOST_AVAILABLE,
    CATBOOST_AVAILABLE,
)


class TestLightGBMWrapper:
    """Test LightGBM wrapper functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=100, 
            n_features=10, 
            n_targets=3,
            noise=0.1, 
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def single_target_data(self):
        """Create single-target regression data."""
        X, y = make_regression(
            n_samples=100, 
            n_features=10, 
            n_targets=1,
            noise=0.1, 
            random_state=42
        )
        return X, y.ravel()
    
    @pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not available")
    def test_lightgbm_initialization(self):
        """Test LightGBM wrapper initialization."""
        model = LightGBMWrapper(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        assert model.n_estimators == 50
        assert model.learning_rate == 0.1
        assert model.max_depth == 5
        assert model.random_state == 42
        assert model.models_ == {}
        assert model.n_targets_ is None
    
    @pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not available")
    def test_lightgbm_single_target_fit_predict(self, single_target_data):
        """Test LightGBM wrapper with single target."""
        X, y = single_target_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LightGBMWrapper(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        assert model.n_targets_ == 1
        assert 0 in model.models_
        
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert not np.isnan(predictions).all()
    
    @pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not available")
    def test_lightgbm_multi_target_fit_predict(self, sample_data):
        """Test LightGBM wrapper with multiple targets."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LightGBMWrapper(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        assert model.n_targets_ == 3
        assert len(model.models_) <= 3  # Some targets might be skipped if all NaN
        
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test), 3)
    
    @pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not available")
    def test_lightgbm_missing_targets(self):
        """Test LightGBM wrapper with missing target values."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 2)
        y[:, 1] = np.nan  # Make second target all NaN
        
        model = LightGBMWrapper(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        assert model.n_targets_ == 2
        assert 0 in model.models_  # First target should be trained
        assert 1 not in model.models_  # Second target should be skipped
        
        predictions = model.predict(X)
        assert predictions.shape == (50, 2)
        assert not np.isnan(predictions[:, 0]).all()  # First target predictions
        assert np.isnan(predictions[:, 1]).all()  # Second target should be NaN
    
    def test_lightgbm_not_available(self):
        """Test error when LightGBM is not available."""
        with patch('src.polymer_prediction.models.ensemble.LIGHTGBM_AVAILABLE', False):
            with pytest.raises(ImportError, match="LightGBM is not available"):
                LightGBMWrapper()


class TestXGBoostWrapper:
    """Test XGBoost wrapper functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=100, 
            n_features=10, 
            n_targets=3,
            noise=0.1, 
            random_state=42
        )
        return X, y
    
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_initialization(self):
        """Test XGBoost wrapper initialization."""
        model = XGBoostWrapper(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        assert model.n_estimators == 50
        assert model.learning_rate == 0.1
        assert model.max_depth == 5
        assert model.random_state == 42
        assert model.models_ == {}
        assert model.n_targets_ is None
    
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_fit_predict(self, sample_data):
        """Test XGBoost wrapper fit and predict."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = XGBoostWrapper(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        assert model.n_targets_ == 3
        assert len(model.models_) <= 3
        
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test), 3)
    
    def test_xgboost_not_available(self):
        """Test error when XGBoost is not available."""
        with patch('src.polymer_prediction.models.ensemble.XGBOOST_AVAILABLE', False):
            with pytest.raises(ImportError, match="XGBoost is not available"):
                XGBoostWrapper()


class TestCatBoostWrapper:
    """Test CatBoost wrapper functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=100, 
            n_features=10, 
            n_targets=3,
            noise=0.1, 
            random_state=42
        )
        return X, y
    
    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_initialization(self):
        """Test CatBoost wrapper initialization."""
        model = CatBoostWrapper(
            iterations=50,
            learning_rate=0.1,
            depth=5,
            random_state=42,
            cat_features=[0, 1]
        )
        
        assert model.iterations == 50
        assert model.learning_rate == 0.1
        assert model.depth == 5
        assert model.random_state == 42
        assert model.cat_features == [0, 1]
        assert model.models_ == {}
        assert model.n_targets_ is None
    
    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_catboost_fit_predict(self, sample_data):
        """Test CatBoost wrapper fit and predict."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = CatBoostWrapper(iterations=10, random_state=42)
        model.fit(X_train, y_train)
        
        assert model.n_targets_ == 3
        assert len(model.models_) <= 3
        
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test), 3)
    
    def test_catboost_not_available(self):
        """Test error when CatBoost is not available."""
        with patch('src.polymer_prediction.models.ensemble.CATBOOST_AVAILABLE', False):
            with pytest.raises(ImportError, match="CatBoost is not available"):
                CatBoostWrapper()


class TestHyperparameterOptimizer:
    """Test hyperparameter optimization functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=50,  # Smaller dataset for faster testing
            n_features=5, 
            n_targets=2,
            noise=0.1, 
            random_state=42
        )
        return X, y
    
    def test_optimizer_initialization(self):
        """Test hyperparameter optimizer initialization."""
        optimizer = HyperparameterOptimizer(
            n_trials=10,
            cv_folds=3,
            random_state=42,
            timeout=60
        )
        
        assert optimizer.n_trials == 10
        assert optimizer.cv_folds == 3
        assert optimizer.random_state == 42
        assert optimizer.timeout == 60
    
    @pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not available")
    def test_optimize_lightgbm(self, sample_data):
        """Test LightGBM hyperparameter optimization."""
        X, y = sample_data
        
        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=2, random_state=42)
        best_params = optimizer.optimize_lightgbm(X, y)
        
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params
        assert 'learning_rate' in best_params
        assert 'max_depth' in best_params
    
    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_optimize_xgboost(self, sample_data):
        """Test XGBoost hyperparameter optimization."""
        X, y = sample_data
        
        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=2, random_state=42)
        best_params = optimizer.optimize_xgboost(X, y)
        
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params
        assert 'learning_rate' in best_params
        assert 'max_depth' in best_params
    
    @pytest.mark.skipif(not CATBOOST_AVAILABLE, reason="CatBoost not available")
    def test_optimize_catboost(self, sample_data):
        """Test CatBoost hyperparameter optimization."""
        X, y = sample_data
        
        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=2, random_state=42)
        best_params = optimizer.optimize_catboost(X, y)
        
        assert isinstance(best_params, dict)
        assert 'iterations' in best_params
        assert 'learning_rate' in best_params
        assert 'depth' in best_params


class TestTreeEnsemble:
    """Test tree ensemble functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample regression data."""
        X, y = make_regression(
            n_samples=50,  # Smaller dataset for faster testing
            n_features=5, 
            n_targets=2,
            noise=0.1, 
            random_state=42
        )
        return X, y
    
    def test_ensemble_initialization(self):
        """Test tree ensemble initialization."""
        ensemble = TreeEnsemble(
            models=['lgbm', 'xgb'],
            optimize_hyperparams=False,
            random_state=42
        )
        
        assert ensemble.models == ['lgbm', 'xgb']
        assert not ensemble.optimize_hyperparams
        assert ensemble.random_state == 42
        assert ensemble.trained_models == {}
        assert ensemble.model_weights == {}
    
    def test_ensemble_auto_model_selection(self):
        """Test automatic model selection based on availability."""
        ensemble = TreeEnsemble()
        
        expected_models = []
        if LIGHTGBM_AVAILABLE:
            expected_models.append('lgbm')
        if XGBOOST_AVAILABLE:
            expected_models.append('xgb')
        if CATBOOST_AVAILABLE:
            expected_models.append('catboost')
        
        assert ensemble.models == expected_models
    
    def test_default_params(self):
        """Test default parameter retrieval."""
        ensemble = TreeEnsemble(random_state=42)
        
        lgbm_params = ensemble._get_default_params('lgbm')
        assert lgbm_params['random_state'] == 42
        assert 'n_estimators' in lgbm_params
        
        xgb_params = ensemble._get_default_params('xgb')
        assert xgb_params['random_state'] == 42
        assert 'n_estimators' in xgb_params
        
        catboost_params = ensemble._get_default_params('catboost')
        assert catboost_params['random_state'] == 42
        assert 'iterations' in catboost_params
    
    @pytest.mark.skipif(
        not (LIGHTGBM_AVAILABLE and XGBOOST_AVAILABLE), 
        reason="Both LightGBM and XGBoost required"
    )
    def test_ensemble_fit_predict(self, sample_data):
        """Test ensemble training and prediction."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        ensemble = TreeEnsemble(
            models=['lgbm', 'xgb'],
            optimize_hyperparams=False,
            random_state=42
        )
        
        ensemble.fit(X_train, y_train)
        
        assert len(ensemble.trained_models) <= 2
        assert len(ensemble.model_weights) <= 2
        
        # Check that weights sum to 1
        total_weight = sum(ensemble.model_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
        
        predictions = ensemble.predict(X_test)
        assert predictions.shape == (len(X_test), 2)
    
    def test_ensemble_no_models_available(self):
        """Test ensemble behavior when no models are available."""
        with patch('src.polymer_prediction.models.ensemble.LIGHTGBM_AVAILABLE', False), \
             patch('src.polymer_prediction.models.ensemble.XGBOOST_AVAILABLE', False), \
             patch('src.polymer_prediction.models.ensemble.CATBOOST_AVAILABLE', False):
            
            ensemble = TreeEnsemble()
            assert ensemble.models == []
    
    def test_ensemble_predict_without_fit(self):
        """Test ensemble prediction without fitting."""
        ensemble = TreeEnsemble(models=['lgbm'])
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="Ensemble must be fitted"):
            ensemble.predict(X)
    
    def test_get_model_info(self, sample_data):
        """Test model information retrieval."""
        if not (LIGHTGBM_AVAILABLE or XGBOOST_AVAILABLE):
            pytest.skip("No tree models available")
        
        X, y = sample_data
        
        available_models = []
        if LIGHTGBM_AVAILABLE:
            available_models.append('lgbm')
        if XGBOOST_AVAILABLE:
            available_models.append('xgb')
        
        ensemble = TreeEnsemble(
            models=available_models[:1],  # Use only first available model
            optimize_hyperparams=False,
            random_state=42
        )
        
        ensemble.fit(X, y)
        info = ensemble.get_model_info()
        
        assert isinstance(info, dict)
        for model_type in ensemble.trained_models.keys():
            assert model_type in info
            assert 'weight' in info[model_type]
            assert 'n_targets' in info[model_type]
            assert 'available_targets' in info[model_type]


class TestIntegration:
    """Integration tests for the complete ensemble pipeline."""
    
    @pytest.fixture
    def polymer_like_data(self):
        """Create data similar to polymer prediction task."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        # Create features (molecular descriptors)
        X = np.random.randn(n_samples, n_features)
        
        # Create 5 targets (polymer properties) with some missing values
        y = np.random.randn(n_samples, 5)
        
        # Introduce missing values in targets (realistic for polymer data)
        missing_mask = np.random.random((n_samples, 5)) < 0.2
        y[missing_mask] = np.nan
        
        return X, y
    
    @pytest.mark.skipif(
        not (LIGHTGBM_AVAILABLE or XGBOOST_AVAILABLE or CATBOOST_AVAILABLE),
        reason="At least one tree model required"
    )
    def test_complete_pipeline(self, polymer_like_data):
        """Test complete ensemble pipeline with polymer-like data."""
        X, y = polymer_like_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create ensemble with available models
        available_models = []
        if LIGHTGBM_AVAILABLE:
            available_models.append('lgbm')
        if XGBOOST_AVAILABLE:
            available_models.append('xgb')
        if CATBOOST_AVAILABLE:
            available_models.append('catboost')
        
        ensemble = TreeEnsemble(
            models=available_models,
            optimize_hyperparams=False,  # Skip optimization for speed
            random_state=42
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Make predictions
        predictions = ensemble.predict(X_test)
        
        # Verify predictions
        assert predictions.shape == (len(X_test), 5)
        assert not np.isnan(predictions).all()  # Should have some valid predictions
        
        # Get model information
        info = ensemble.get_model_info()
        assert len(info) > 0
        
        # Verify weights sum to 1
        total_weight = sum(w['weight'] for w in info.values())
        assert abs(total_weight - 1.0) < 1e-6