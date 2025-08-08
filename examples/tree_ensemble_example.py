"""
Example usage of tree ensemble models for polymer property prediction.

This script demonstrates how to use the LightGBM, XGBoost, and CatBoost
wrappers with hyperparameter optimization and ensemble combination.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.polymer_prediction.models.ensemble import (
    LightGBMWrapper,
    XGBoostWrapper,
    CatBoostWrapper,
    HyperparameterOptimizer,
    TreeEnsemble,
)


def create_sample_polymer_data():
    """Create sample data similar to polymer prediction task."""
    print("Creating sample polymer-like data...")
    
    # Create synthetic data with 5 targets (polymer properties)
    X, y = make_regression(
        n_samples=500,
        n_features=50,  # Molecular descriptors
        n_targets=5,    # Polymer properties (Tg, FFV, Tc, Density, Rg)
        noise=0.1,
        random_state=42
    )
    
    # Introduce some missing values in targets (realistic for polymer data)
    missing_mask = np.random.random(y.shape) < 0.15
    y[missing_mask] = np.nan
    
    return X, y


def example_individual_models():
    """Example of using individual tree models."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Individual Tree Models")
    print("="*60)
    
    X, y = create_sample_polymer_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training targets shape: {y_train.shape}")
    
    # Example 1: LightGBM
    print("\n1. Training LightGBM model...")
    lgbm_model = LightGBMWrapper(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    lgbm_model.fit(X_train, y_train)
    lgbm_pred = lgbm_model.predict(X_test)
    
    # Calculate RMSE for each target
    for target_idx in range(y_test.shape[1]):
        mask = ~np.isnan(y_test[:, target_idx])
        if np.any(mask):
            rmse = np.sqrt(mean_squared_error(
                y_test[mask, target_idx], 
                lgbm_pred[mask, target_idx]
            ))
            print(f"   Target {target_idx} RMSE: {rmse:.4f}")
    
    # Example 2: XGBoost
    print("\n2. Training XGBoost model...")
    xgb_model = XGBoostWrapper(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    for target_idx in range(y_test.shape[1]):
        mask = ~np.isnan(y_test[:, target_idx])
        if np.any(mask):
            rmse = np.sqrt(mean_squared_error(
                y_test[mask, target_idx], 
                xgb_pred[mask, target_idx]
            ))
            print(f"   Target {target_idx} RMSE: {rmse:.4f}")
    
    # Example 3: CatBoost
    print("\n3. Training CatBoost model...")
    catboost_model = CatBoostWrapper(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        random_state=42
    )
    catboost_model.fit(X_train, y_train)
    catboost_pred = catboost_model.predict(X_test)
    
    for target_idx in range(y_test.shape[1]):
        mask = ~np.isnan(y_test[:, target_idx])
        if np.any(mask):
            rmse = np.sqrt(mean_squared_error(
                y_test[mask, target_idx], 
                catboost_pred[mask, target_idx]
            ))
            print(f"   Target {target_idx} RMSE: {rmse:.4f}")


def example_hyperparameter_optimization():
    """Example of hyperparameter optimization."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Hyperparameter Optimization")
    print("="*60)
    
    X, y = create_sample_polymer_data()
    
    # Use smaller dataset for faster optimization
    X_small = X[:200]
    y_small = y[:200]
    
    print(f"Optimization data shape: {X_small.shape}")
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        n_trials=10,  # Small number for demo
        cv_folds=3,
        random_state=42
    )
    
    # Optimize LightGBM
    print("\n1. Optimizing LightGBM hyperparameters...")
    lgbm_best_params = optimizer.optimize_lightgbm(X_small, y_small)
    print("   Best LightGBM parameters:")
    for param, value in lgbm_best_params.items():
        print(f"     {param}: {value}")
    
    # Optimize XGBoost
    print("\n2. Optimizing XGBoost hyperparameters...")
    xgb_best_params = optimizer.optimize_xgboost(X_small, y_small)
    print("   Best XGBoost parameters:")
    for param, value in xgb_best_params.items():
        print(f"     {param}: {value}")
    
    # Optimize CatBoost
    print("\n3. Optimizing CatBoost hyperparameters...")
    catboost_best_params = optimizer.optimize_catboost(X_small, y_small)
    print("   Best CatBoost parameters:")
    for param, value in catboost_best_params.items():
        print(f"     {param}: {value}")


def example_tree_ensemble():
    """Example of complete tree ensemble."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Complete Tree Ensemble")
    print("="*60)
    
    X, y = create_sample_polymer_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training targets shape: {y_train.shape}")
    
    # Create ensemble with all available models
    print("\n1. Creating tree ensemble...")
    ensemble = TreeEnsemble(
        models=['lgbm', 'xgb', 'catboost'],
        optimize_hyperparams=False,  # Skip optimization for demo speed
        random_state=42
    )
    
    # Train ensemble
    print("\n2. Training ensemble models...")
    ensemble.fit(X_train, y_train)
    
    # Get model information
    print("\n3. Model information:")
    model_info = ensemble.get_model_info()
    for model_type, info in model_info.items():
        print(f"   {model_type}:")
        print(f"     Weight: {info['weight']:.4f}")
        print(f"     Targets: {len(info['available_targets'])}")
    
    # Make predictions
    print("\n4. Making ensemble predictions...")
    ensemble_pred = ensemble.predict(X_test)
    
    # Calculate ensemble RMSE
    print("\n5. Ensemble performance:")
    for target_idx in range(y_test.shape[1]):
        mask = ~np.isnan(y_test[:, target_idx])
        if np.any(mask):
            rmse = np.sqrt(mean_squared_error(
                y_test[mask, target_idx], 
                ensemble_pred[mask, target_idx]
            ))
            print(f"   Target {target_idx} RMSE: {rmse:.4f}")


def example_with_optimization():
    """Example of ensemble with hyperparameter optimization."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Ensemble with Hyperparameter Optimization")
    print("="*60)
    
    X, y = create_sample_polymer_data()
    
    # Use smaller dataset for faster optimization
    X_small = X[:300]
    y_small = y[:300]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.2, random_state=42
    )
    
    print(f"Training data shape: {X_train.shape}")
    
    # Create ensemble with optimization
    print("\n1. Creating optimized tree ensemble...")
    ensemble = TreeEnsemble(
        models=['lgbm', 'xgb'],  # Use fewer models for faster demo
        optimize_hyperparams=True,
        n_trials=5,  # Small number for demo
        cv_folds=3,
        random_state=42
    )
    
    # Train ensemble (this will include hyperparameter optimization)
    print("\n2. Training ensemble with optimization...")
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    print("\n3. Making optimized predictions...")
    optimized_pred = ensemble.predict(X_test)
    
    # Calculate performance
    print("\n4. Optimized ensemble performance:")
    for target_idx in range(y_test.shape[1]):
        mask = ~np.isnan(y_test[:, target_idx])
        if np.any(mask):
            rmse = np.sqrt(mean_squared_error(
                y_test[mask, target_idx], 
                optimized_pred[mask, target_idx]
            ))
            print(f"   Target {target_idx} RMSE: {rmse:.4f}")
    
    # Show model weights
    model_info = ensemble.get_model_info()
    print("\n5. Optimized model weights:")
    for model_type, info in model_info.items():
        print(f"   {model_type}: {info['weight']:.4f}")


if __name__ == "__main__":
    print("Tree Ensemble Models Example")
    print("="*60)
    
    try:
        # Run examples
        example_individual_models()
        example_hyperparameter_optimization()
        example_tree_ensemble()
        example_with_optimization()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()