"""
Tree ensemble models for polymer property prediction.

This module implements LightGBM, XGBoost, and CatBoost models with
hyperparameter optimization using Optuna.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import optuna
from optuna.samplers import TPESampler

# Tree ensemble imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

logger = logging.getLogger(__name__)


class LightGBMWrapper(BaseEstimator, RegressorMixin):
    """
    LightGBM model wrapper with proper parameter handling and training methods.
    
    This wrapper provides a scikit-learn compatible interface for LightGBM
    with support for multi-target regression and hyperparameter optimization.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: int = 42,
        verbose: int = -1,
        **kwargs
    ):
        """
        Initialize LightGBM wrapper.
        
        Args:
            n_estimators: Number of boosting iterations
            learning_rate: Learning rate
            max_depth: Maximum tree depth (-1 means no limit)
            num_leaves: Maximum number of leaves in one tree
            min_child_samples: Minimum number of data needed in a child
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            random_state: Random seed
            verbose: Verbosity level
            **kwargs: Additional parameters for LightGBM
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available. Install with: pip install lightgbm")
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs
        
        self.models_ = {}
        self.n_targets_ = None
        
    def _get_params(self) -> Dict[str, Any]:
        """Get parameters for LightGBM model."""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'verbose': self.verbose,
            **self.kwargs
        }
        return params
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMWrapper':
        """
        Fit LightGBM model(s) to training data.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,) or (n_samples, n_targets)
            
        Returns:
            self: Fitted estimator
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.n_targets_ = y.shape[1]
        params = self._get_params()
        
        # Train separate model for each target
        for target_idx in range(self.n_targets_):
            y_target = y[:, target_idx]
            
            # Handle missing values by creating a mask
            mask = ~np.isnan(y_target)
            if not np.any(mask):
                logger.warning(f"No valid targets for target {target_idx}, skipping")
                continue
                
            X_masked = X[mask]
            y_masked = y_target[mask]
            
            # Create LightGBM dataset
            train_data = lgb.Dataset(X_masked, label=y_masked)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                valid_sets=[train_data],
                callbacks=[lgb.log_evaluation(0)]  # Suppress training logs
            )
            
            self.models_[target_idx] = model
            
        logger.info(f"Trained LightGBM models for {len(self.models_)} targets")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted LightGBM model(s).
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_targets)
        """
        if not self.models_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        if self.n_targets_ == 1:
            # Single target
            if 0 in self.models_:
                predictions = self.models_[0].predict(X)
            else:
                predictions = np.full(n_samples, np.nan)
            return predictions
        else:
            # Multi-target
            predictions = np.full((n_samples, self.n_targets_), np.nan)
            for target_idx, model in self.models_.items():
                predictions[:, target_idx] = model.predict(X)
            return predictions


class XGBoostWrapper(BaseEstimator, RegressorMixin):
    """
    XGBoost model wrapper with hyperparameter optimization support.
    
    This wrapper provides a scikit-learn compatible interface for XGBoost
    with support for multi-target regression and hyperparameter optimization.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: int = 1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize XGBoost wrapper.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum depth of a tree
            min_child_weight: Minimum sum of instance weight needed in a child
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            random_state: Random seed
            **kwargs: Additional parameters for XGBoost
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.models_ = {}
        self.n_targets_ = None
        
    def _get_params(self) -> Dict[str, Any]:
        """Get parameters for XGBoost model."""
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'verbosity': 0,  # Suppress training logs
            **self.kwargs
        }
        return params
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'XGBoostWrapper':
        """
        Fit XGBoost model(s) to training data.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,) or (n_samples, n_targets)
            
        Returns:
            self: Fitted estimator
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.n_targets_ = y.shape[1]
        params = self._get_params()
        
        # Train separate model for each target
        for target_idx in range(self.n_targets_):
            y_target = y[:, target_idx]
            
            # Handle missing values by creating a mask
            mask = ~np.isnan(y_target)
            if not np.any(mask):
                logger.warning(f"No valid targets for target {target_idx}, skipping")
                continue
                
            X_masked = X[mask]
            y_masked = y_target[mask]
            
            # Create and train XGBoost model
            model = xgb.XGBRegressor(**params)
            model.fit(X_masked, y_masked)
            
            self.models_[target_idx] = model
            
        logger.info(f"Trained XGBoost models for {len(self.models_)} targets")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted XGBoost model(s).
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_targets)
        """
        if not self.models_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        if self.n_targets_ == 1:
            # Single target
            if 0 in self.models_:
                predictions = self.models_[0].predict(X)
            else:
                predictions = np.full(n_samples, np.nan)
            return predictions
        else:
            # Multi-target
            predictions = np.full((n_samples, self.n_targets_), np.nan)
            for target_idx, model in self.models_.items():
                predictions[:, target_idx] = model.predict(X)
            return predictions


class CatBoostWrapper(BaseEstimator, RegressorMixin):
    """
    CatBoost model wrapper with proper categorical feature handling.
    
    This wrapper provides a scikit-learn compatible interface for CatBoost
    with support for multi-target regression and categorical features.
    """
    
    def __init__(
        self,
        iterations: int = 100,
        learning_rate: float = 0.1,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        subsample: float = 1.0,
        colsample_bylevel: float = 1.0,
        random_state: int = 42,
        cat_features: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Initialize CatBoost wrapper.
        
        Args:
            iterations: Number of boosting iterations
            learning_rate: Learning rate
            depth: Depth of the tree
            l2_leaf_reg: L2 regularization coefficient
            subsample: Sample rate for bagging
            colsample_bylevel: Feature sampling rate
            random_state: Random seed
            cat_features: List of categorical feature indices
            **kwargs: Additional parameters for CatBoost
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not available. Install with: pip install catboost")
        
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.random_state = random_state
        self.cat_features = cat_features or []
        self.kwargs = kwargs
        
        self.models_ = {}
        self.n_targets_ = None
        
    def _get_params(self) -> Dict[str, Any]:
        """Get parameters for CatBoost model."""
        params = {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'subsample': self.subsample,
            'colsample_bylevel': self.colsample_bylevel,
            'random_state': self.random_state,
            'verbose': False,  # Suppress training logs
            **self.kwargs
        }
        return params
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CatBoostWrapper':
        """
        Fit CatBoost model(s) to training data.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,) or (n_samples, n_targets)
            
        Returns:
            self: Fitted estimator
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.n_targets_ = y.shape[1]
        params = self._get_params()
        
        # Train separate model for each target
        for target_idx in range(self.n_targets_):
            y_target = y[:, target_idx]
            
            # Handle missing values by creating a mask
            mask = ~np.isnan(y_target)
            if not np.any(mask):
                logger.warning(f"No valid targets for target {target_idx}, skipping")
                continue
                
            X_masked = X[mask]
            y_masked = y_target[mask]
            
            # Create and train CatBoost model
            model = cb.CatBoostRegressor(**params)
            model.fit(
                X_masked, 
                y_masked,
                cat_features=self.cat_features,
                verbose=False
            )
            
            self.models_[target_idx] = model
            
        logger.info(f"Trained CatBoost models for {len(self.models_)} targets")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted CatBoost model(s).
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,) or (n_samples, n_targets)
        """
        if not self.models_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        if self.n_targets_ == 1:
            # Single target
            if 0 in self.models_:
                predictions = self.models_[0].predict(X)
            else:
                predictions = np.full(n_samples, np.nan)
            return predictions
        else:
            # Multi-target
            predictions = np.full((n_samples, self.n_targets_), np.nan)
            for target_idx, model in self.models_.items():
                predictions[:, target_idx] = model.predict(X)
            return predictions


class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter optimization for tree ensemble models.
    
    This class provides hyperparameter optimization with proper objective functions
    for each model type (LightGBM, XGBoost, CatBoost).
    """
    
    def __init__(
        self,
        n_trials: int = 100,
        cv_folds: int = 5,
        random_state: int = 42,
        timeout: Optional[int] = None
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            timeout: Timeout in seconds for optimization
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.timeout = timeout
        
    def _objective_lightgbm(
        self, 
        trial: optuna.Trial, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> float:
        """
        Objective function for LightGBM hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X: Training features
            y: Training targets
            
        Returns:
            Cross-validation RMSE score
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': self.random_state,
            'verbose': -1
        }
        
        model = LightGBMWrapper(**params)
        
        # Perform cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Handle multi-target case
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)
                y_val = y_val.reshape(-1, 1)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            
            # Calculate RMSE for each target and average
            target_scores = []
            for target_idx in range(y_val.shape[1]):
                mask = ~np.isnan(y_val[:, target_idx])
                if np.any(mask):
                    score = np.sqrt(mean_squared_error(
                        y_val[mask, target_idx], 
                        y_pred[mask, target_idx]
                    ))
                    target_scores.append(score)
            
            if target_scores:
                scores.append(np.mean(target_scores))
        
        return np.mean(scores) if scores else float('inf')
    
    def _objective_xgboost(
        self, 
        trial: optuna.Trial, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> float:
        """
        Objective function for XGBoost hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X: Training features
            y: Training targets
            
        Returns:
            Cross-validation RMSE score
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': self.random_state
        }
        
        model = XGBoostWrapper(**params)
        
        # Perform cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Handle multi-target case
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)
                y_val = y_val.reshape(-1, 1)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            
            # Calculate RMSE for each target and average
            target_scores = []
            for target_idx in range(y_val.shape[1]):
                mask = ~np.isnan(y_val[:, target_idx])
                if np.any(mask):
                    score = np.sqrt(mean_squared_error(
                        y_val[mask, target_idx], 
                        y_pred[mask, target_idx]
                    ))
                    target_scores.append(score)
            
            if target_scores:
                scores.append(np.mean(target_scores))
        
        return np.mean(scores) if scores else float('inf')
    
    def _objective_catboost(
        self, 
        trial: optuna.Trial, 
        X: np.ndarray, 
        y: np.ndarray,
        cat_features: Optional[List[int]] = None
    ) -> float:
        """
        Objective function for CatBoost hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X: Training features
            y: Training targets
            cat_features: List of categorical feature indices
            
        Returns:
            Cross-validation RMSE score
        """
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'random_state': self.random_state,
            'cat_features': cat_features or []
        }
        
        model = CatBoostWrapper(**params)
        
        # Perform cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Handle multi-target case
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)
                y_val = y_val.reshape(-1, 1)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            
            # Calculate RMSE for each target and average
            target_scores = []
            for target_idx in range(y_val.shape[1]):
                mask = ~np.isnan(y_val[:, target_idx])
                if np.any(mask):
                    score = np.sqrt(mean_squared_error(
                        y_val[mask, target_idx], 
                        y_pred[mask, target_idx]
                    ))
                    target_scores.append(score)
            
            if target_scores:
                scores.append(np.mean(target_scores))
        
        return np.mean(scores) if scores else float('inf')
    
    def optimize_lightgbm(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Best hyperparameters
        """
        logger.info("Starting LightGBM hyperparameter optimization...")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            lambda trial: self._objective_lightgbm(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        logger.info(f"LightGBM optimization completed. Best score: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_xgboost(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Best hyperparameters
        """
        logger.info("Starting XGBoost hyperparameter optimization...")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            lambda trial: self._objective_xgboost(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        logger.info(f"XGBoost optimization completed. Best score: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_catboost(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        cat_features: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Optimize CatBoost hyperparameters.
        
        Args:
            X: Training features
            y: Training targets
            cat_features: List of categorical feature indices
            
        Returns:
            Best hyperparameters
        """
        logger.info("Starting CatBoost hyperparameter optimization...")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            lambda trial: self._objective_catboost(trial, X, y, cat_features),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        logger.info(f"CatBoost optimization completed. Best score: {study.best_value:.4f}")
        return study.best_params


class TreeEnsemble:
    """
    Complete tree ensemble that trains LightGBM, XGBoost, and CatBoost models
    and combines their predictions.
    
    This class provides a unified interface for training multiple tree-based models
    with hyperparameter optimization and ensemble combination.
    """
    
    def __init__(
        self,
        models: List[str] = None,
        optimize_hyperparams: bool = True,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        cat_features: Optional[List[int]] = None
    ):
        """
        Initialize tree ensemble.
        
        Args:
            models: List of model types to include ('lgbm', 'xgb', 'catboost')
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials per model
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            cat_features: List of categorical feature indices for CatBoost
        """
        if models is None:
            models = []
            if LIGHTGBM_AVAILABLE:
                models.append('lgbm')
            if XGBOOST_AVAILABLE:
                models.append('xgb')
            if CATBOOST_AVAILABLE:
                models.append('catboost')
        
        self.models = models
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cat_features = cat_features or []
        
        self.trained_models = {}
        self.model_weights = {}
        self.optimizer = None
        
        if self.optimize_hyperparams:
            self.optimizer = HyperparameterOptimizer(
                n_trials=n_trials,
                cv_folds=cv_folds,
                random_state=random_state
            )
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for each model type."""
        defaults = {
            'lgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': -1,
                'num_leaves': 31,
                'random_state': self.random_state
            },
            'xgb': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': self.random_state
            },
            'catboost': {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'random_state': self.random_state,
                'cat_features': self.cat_features
            }
        }
        return defaults.get(model_type, {})
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TreeEnsemble':
        """
        Train all tree ensemble models.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,) or (n_samples, n_targets)
            
        Returns:
            self: Fitted ensemble
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        logger.info(f"Training tree ensemble with models: {self.models}")
        
        for model_type in self.models:
            logger.info(f"Training {model_type} model...")
            
            try:
                # Get optimized or default parameters
                if self.optimize_hyperparams and self.optimizer:
                    if model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
                        params = self.optimizer.optimize_lightgbm(X, y)
                    elif model_type == 'xgb' and XGBOOST_AVAILABLE:
                        params = self.optimizer.optimize_xgboost(X, y)
                    elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                        params = self.optimizer.optimize_catboost(X, y, self.cat_features)
                    else:
                        params = self._get_default_params(model_type)
                else:
                    params = self._get_default_params(model_type)
                
                # Create and train model
                if model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
                    model = LightGBMWrapper(**params)
                elif model_type == 'xgb' and XGBOOST_AVAILABLE:
                    model = XGBoostWrapper(**params)
                elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                    model = CatBoostWrapper(**params)
                else:
                    logger.warning(f"Model {model_type} not available, skipping")
                    continue
                
                model.fit(X, y)
                self.trained_models[model_type] = model
                
                # Calculate model weight based on cross-validation performance
                cv_score = self._calculate_cv_score(model, X, y)
                self.model_weights[model_type] = 1.0 / (cv_score + 1e-8)  # Inverse of error
                
                logger.info(f"{model_type} model trained successfully (CV score: {cv_score:.4f})")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} model: {str(e)}")
                continue
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {
                k: v / total_weight for k, v in self.model_weights.items()
            }
        
        logger.info(f"Tree ensemble training completed. Model weights: {self.model_weights}")
        return self
    
    def _calculate_cv_score(
        self, 
        model: Union[LightGBMWrapper, XGBoostWrapper, CatBoostWrapper],
        X: np.ndarray, 
        y: np.ndarray
    ) -> float:
        """Calculate cross-validation score for model weighting."""
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create a copy of the model for CV
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_val)
            
            # Handle multi-target case
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            
            # Calculate RMSE for each target and average
            target_scores = []
            for target_idx in range(y_val.shape[1]):
                mask = ~np.isnan(y_val[:, target_idx])
                if np.any(mask):
                    score = np.sqrt(mean_squared_error(
                        y_val[mask, target_idx], 
                        y_pred[mask, target_idx]
                    ))
                    target_scores.append(score)
            
            if target_scores:
                scores.append(np.mean(target_scores))
        
        return np.mean(scores) if scores else float('inf')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble of trained models.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Weighted ensemble predictions
        """
        if not self.trained_models:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # Get predictions from all models
        model_predictions = {}
        for model_type, model in self.trained_models.items():
            try:
                pred = model.predict(X)
                model_predictions[model_type] = pred
            except Exception as e:
                logger.warning(f"Failed to get predictions from {model_type}: {str(e)}")
                continue
        
        if not model_predictions:
            raise ValueError("No models produced valid predictions")
        
        # Determine output shape
        sample_pred = next(iter(model_predictions.values()))
        if sample_pred.ndim == 1:
            n_targets = 1
            ensemble_pred = np.zeros(n_samples)
        else:
            n_targets = sample_pred.shape[1]
            ensemble_pred = np.zeros((n_samples, n_targets))
        
        # Weighted combination of predictions
        total_weight = 0.0
        for model_type, pred in model_predictions.items():
            weight = self.model_weights.get(model_type, 1.0)
            
            if pred.ndim == 1 and n_targets > 1:
                pred = pred.reshape(-1, 1)
            elif pred.ndim == 2 and n_targets == 1:
                pred = pred.flatten()
            
            ensemble_pred += weight * pred
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        logger.info(f"Generated ensemble predictions using {len(model_predictions)} models")
        return ensemble_pred
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about trained models.
        
        Returns:
            Dictionary with model information
        """
        info = {}
        for model_type, model in self.trained_models.items():
            info[model_type] = {
                'weight': self.model_weights.get(model_type, 0.0),
                'n_targets': getattr(model, 'n_targets_', None),
                'available_targets': list(getattr(model, 'models_', {}).keys())
            }
        return info