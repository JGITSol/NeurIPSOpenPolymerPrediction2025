"""
Stacking Ensemble with Cross-Validation for Polymer Prediction

This module implements a stacking ensemble that combines GCN and tree ensemble models
using cross-validation to generate out-of-fold predictions and a meta-learner to
combine base model predictions.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import gc

from ..training.trainer import train_one_epoch, evaluate, predict, masked_mse_loss
from .ensemble import TreeEnsemble
from ..data.dataset import PolymerDataset

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble that combines GCN and tree ensemble models using cross-validation.
    
    This class implements a two-level stacking approach:
    1. Base models (GCN and tree ensemble) generate out-of-fold predictions using CV
    2. Meta-learner (Ridge regression) learns to combine base model predictions
    """
    
    def __init__(
        self,
        gcn_model_class,
        gcn_params: Dict[str, Any] = None,
        tree_models: List[str] = None,
        tree_params: Dict[str, Any] = None,
        meta_model = None,
        cv_folds: int = 5,
        random_state: int = 42,
        device: torch.device = None,
        batch_size: int = 32,
        gcn_epochs: int = 50,
        optimize_tree_hyperparams: bool = False,
        tree_optuna_trials: int = 20
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            gcn_model_class: Class for GCN model
            gcn_params: Parameters for GCN model initialization
            tree_models: List of tree model types ('lgbm', 'xgb', 'catboost')
            tree_params: Parameters for tree ensemble
            meta_model: Meta-learner model (default: Ridge regression)
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            device: PyTorch device for GCN training
            batch_size: Batch size for GCN training
            gcn_epochs: Number of epochs for GCN training
            optimize_tree_hyperparams: Whether to optimize tree model hyperparameters
            tree_optuna_trials: Number of Optuna trials for tree optimization
        """
        self.gcn_model_class = gcn_model_class
        self.gcn_params = gcn_params or {}
        self.tree_models = tree_models or ['lgbm', 'xgb', 'catboost']
        self.tree_params = tree_params or {}
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.device = device or torch.device('cpu')
        self.batch_size = batch_size
        self.gcn_epochs = gcn_epochs
        self.optimize_tree_hyperparams = optimize_tree_hyperparams
        self.tree_optuna_trials = tree_optuna_trials
        
        # Fitted components
        self.base_models_ = {}
        self.meta_models_ = {}
        self.feature_extractor_ = None
        self.n_targets_ = None
        self.target_cols_ = None
        self.cv_scores_ = {}
        
        # Set random seeds
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
    
    def _create_cv_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation splits ensuring no data leakage.
        
        Args:
            X: Feature matrix
            y: Target matrix
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        # Use KFold for regression tasks
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Create splits based on samples, not individual targets
        splits = list(kf.split(X))
        
        logger.info(f"Created {len(splits)} CV splits with approximately "
                   f"{len(splits[0][0])} train and {len(splits[0][1])} validation samples each")
        
        return splits
    
    def _train_gcn_fold(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame,
        fold_idx: int
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Train GCN model on one CV fold.
        
        Args:
            train_df: Training data for this fold
            val_df: Validation data for this fold
            fold_idx: Fold index for logging
            
        Returns:
            Tuple of (validation predictions, validation scores)
        """
        logger.info(f"Training GCN for fold {fold_idx + 1}/{self.cv_folds}")
        
        try:
            # Create datasets
            train_dataset = PolymerDataset(
                train_df, 
                target_cols=self.target_cols_, 
                is_test=False
            )
            val_dataset = PolymerDataset(
                val_df, 
                target_cols=self.target_cols_, 
                is_test=False
            )
            
            # Create data loaders with proper error handling
            from torch_geometric.data import Batch
            
            def collate_fn(batch):
                valid_batch = [item for item in batch if item is not None]
                if len(valid_batch) == 0:
                    return None
                # Use PyG's Batch.from_data_list for proper batching
                return Batch.from_data_list(valid_batch)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                collate_fn=collate_fn
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                collate_fn=collate_fn
            )
            
            # Get number of atom features from a sample
            sample_data = None
            for data in train_dataset:
                if data is not None:
                    sample_data = data
                    break
            
            if sample_data is None:
                raise ValueError(f"No valid molecular graphs found in fold {fold_idx}")
            
            num_atom_features = sample_data.x.size(1)
            
            # Initialize model
            model = self.gcn_model_class(
                num_atom_features=num_atom_features,
                **self.gcn_params
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(self.gcn_epochs):
                # Train
                train_loss = train_one_epoch(model, train_loader, optimizer, self.device)
                
                # Validate
                val_loss, val_rmses = evaluate(model, val_loader, self.device)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Fold {fold_idx + 1}, Epoch {epoch + 1}: "
                              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Load best model state
            model.load_state_dict(best_model_state)
            
            # Generate validation predictions
            try:
                val_ids, val_preds = predict(model, val_loader, self.device)
            except Exception as e:
                logger.error(f"Error in predict function: {str(e)}")
                # Create dummy predictions
                n_val_samples = len(val_df)
                val_preds = np.full((n_val_samples, self.n_targets_), np.nan)
                val_ids = list(range(n_val_samples))
            
            # Calculate final validation scores
            final_val_loss, final_val_rmses = evaluate(model, val_loader, self.device)
            
            # Clean up
            del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return val_preds, final_val_rmses
            
        except Exception as e:
            logger.error(f"Error training GCN for fold {fold_idx}: {str(e)}")
            # Return dummy predictions in case of error
            n_val_samples = len(val_df)
            dummy_preds = np.full((n_val_samples, self.n_targets_), np.nan)
            dummy_scores = {col: np.nan for col in self.target_cols_}
            return dummy_preds, dummy_scores
    
    def _train_tree_fold(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        fold_idx: int
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Train tree ensemble on one CV fold.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            fold_idx: Fold index for logging
            
        Returns:
            Tuple of (validation predictions, validation scores)
        """
        logger.info(f"Training tree ensemble for fold {fold_idx + 1}/{self.cv_folds}")
        
        try:
            # Create tree ensemble
            tree_ensemble = TreeEnsemble(
                models=self.tree_models,
                optimize_hyperparams=self.optimize_tree_hyperparams,
                n_trials=self.tree_optuna_trials,
                random_state=self.random_state,
                **self.tree_params
            )
            
            # Train ensemble
            tree_ensemble.fit(X_train, y_train)
            
            # Generate validation predictions
            val_preds = tree_ensemble.predict(X_val)
            
            # Calculate validation scores
            val_scores = {}
            for i, col in enumerate(self.target_cols_):
                mask = ~np.isnan(y_val[:, i])
                if np.any(mask):
                    rmse = np.sqrt(mean_squared_error(
                        y_val[mask, i], 
                        val_preds[mask, i]
                    ))
                    val_scores[col] = rmse
                else:
                    val_scores[col] = np.nan
            
            return val_preds, val_scores
            
        except Exception as e:
            logger.error(f"Error training tree ensemble for fold {fold_idx}: {str(e)}")
            # Return dummy predictions in case of error
            dummy_preds = np.full((len(X_val), self.n_targets_), np.nan)
            dummy_scores = {col: np.nan for col in self.target_cols_}
            return dummy_preds, dummy_scores
    
    def _generate_base_predictions(
        self, 
        df: pd.DataFrame, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate out-of-fold predictions from base models using cross-validation.
        
        Args:
            df: Full dataframe with SMILES and targets
            X: Feature matrix for tree models
            y: Target matrix
            
        Returns:
            Tuple of (gcn_oof_preds, tree_oof_preds)
        """
        logger.info("Generating out-of-fold predictions from base models...")
        
        # Initialize out-of-fold prediction arrays
        gcn_oof_preds = np.full((len(df), self.n_targets_), np.nan)
        tree_oof_preds = np.full((len(df), self.n_targets_), np.nan)
        
        # Create CV splits
        cv_splits = self._create_cv_splits(X, y)
        
        # Store CV scores for each fold
        gcn_fold_scores = []
        tree_fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            logger.info(f"Processing fold {fold_idx + 1}/{self.cv_folds}")
            
            # Split data
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train GCN and get validation predictions
            gcn_val_preds, gcn_val_scores = self._train_gcn_fold(
                train_df, val_df, fold_idx
            )
            gcn_oof_preds[val_idx] = gcn_val_preds
            gcn_fold_scores.append(gcn_val_scores)
            
            # Train tree ensemble and get validation predictions
            tree_val_preds, tree_val_scores = self._train_tree_fold(
                X_train, y_train, X_val, y_val, fold_idx
            )
            tree_oof_preds[val_idx] = tree_val_preds
            tree_fold_scores.append(tree_val_scores)
            
            logger.info(f"Fold {fold_idx + 1} completed")
        
        # Calculate average CV scores
        self.cv_scores_['gcn'] = self._average_fold_scores(gcn_fold_scores)
        self.cv_scores_['tree'] = self._average_fold_scores(tree_fold_scores)
        
        logger.info("Out-of-fold prediction generation completed")
        logger.info(f"GCN CV scores: {self.cv_scores_['gcn']}")
        logger.info(f"Tree CV scores: {self.cv_scores_['tree']}")
        
        return gcn_oof_preds, tree_oof_preds
    
    def _average_fold_scores(self, fold_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Average scores across folds, handling NaN values."""
        avg_scores = {}
        for col in self.target_cols_:
            scores = [fold_score[col] for fold_score in fold_scores 
                     if not np.isnan(fold_score[col])]
            avg_scores[col] = np.mean(scores) if scores else np.nan
        return avg_scores
    
    def _train_meta_models(
        self, 
        base_predictions: np.ndarray, 
        y: np.ndarray
    ) -> None:
        """
        Train meta-learner models to combine base model predictions.
        
        Args:
            base_predictions: Combined base model predictions (n_samples, n_base_models * n_targets)
            y: True target values (n_samples, n_targets)
        """
        logger.info("Training meta-learner models...")
        
        # Train separate meta-model for each target
        for target_idx, target_col in enumerate(self.target_cols_):
            logger.info(f"Training meta-model for {target_col}")
            
            # Get target values and mask for non-missing values
            y_target = y[:, target_idx]
            mask = ~np.isnan(y_target)
            
            if not np.any(mask):
                logger.warning(f"No valid targets for {target_col}, skipping meta-model training")
                continue
            
            # Get base predictions for this target
            # Assuming base_predictions has shape (n_samples, 2 * n_targets)
            # where first n_targets columns are GCN, next n_targets are tree
            gcn_pred_col = target_idx
            tree_pred_col = self.n_targets_ + target_idx
            
            # Extract predictions for this target from both models
            X_meta = base_predictions[mask][:, [gcn_pred_col, tree_pred_col]]
            y_meta = y_target[mask]
            
            # Handle cases where base predictions might be NaN
            valid_pred_mask = ~np.isnan(X_meta).any(axis=1)
            if not np.any(valid_pred_mask):
                logger.warning(f"No valid base predictions for {target_col}, skipping")
                continue
            
            X_meta = X_meta[valid_pred_mask]
            y_meta = y_meta[valid_pred_mask]
            
            # Train meta-model
            meta_model = Ridge(alpha=1.0, random_state=self.random_state)
            meta_model.fit(X_meta, y_meta)
            
            self.meta_models_[target_idx] = meta_model
            
            # Log meta-model performance
            meta_pred = meta_model.predict(X_meta)
            meta_rmse = np.sqrt(mean_squared_error(y_meta, meta_pred))
            logger.info(f"Meta-model RMSE for {target_col}: {meta_rmse:.4f}")
        
        logger.info("Meta-learner training completed")
    
    def fit(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            df: DataFrame with SMILES and target columns
            X: Feature matrix for tree models (n_samples, n_features)
            y: Target matrix (n_samples, n_targets)
            
        Returns:
            self: Fitted estimator
        """
        logger.info("Starting stacking ensemble training...")
        
        # Store target information
        self.n_targets_ = y.shape[1]
        
        # Try to get target column names from DataFrame
        # Look for common target columns first
        common_targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        available_targets = [col for col in common_targets if col in df.columns]
        
        if len(available_targets) == self.n_targets_:
            self.target_cols_ = available_targets
        else:
            # Fallback to generic names
            self.target_cols_ = [f'target_{i}' for i in range(self.n_targets_)]
        
        # Generate out-of-fold predictions from base models
        gcn_oof_preds, tree_oof_preds = self._generate_base_predictions(df, X, y)
        
        # Combine base predictions for meta-learning
        base_predictions = np.concatenate([gcn_oof_preds, tree_oof_preds], axis=1)
        
        # Train meta-learner models
        self._train_meta_models(base_predictions, y)
        
        # Train final base models on full dataset for prediction
        logger.info("Training final base models on full dataset...")
        
        # Train final GCN model
        try:
            from torch_geometric.data import Batch
            
            def collate_fn(batch):
                valid_batch = [item for item in batch if item is not None]
                if len(valid_batch) == 0:
                    return None
                return Batch.from_data_list(valid_batch)
            
            full_dataset = PolymerDataset(df, target_cols=self.target_cols_, is_test=False)
            full_loader = DataLoader(
                full_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                collate_fn=collate_fn
            )
            
            # Get sample for num_atom_features
            sample_data = None
            for data in full_dataset:
                if data is not None:
                    sample_data = data
                    break
            
            if sample_data is not None:
                num_atom_features = sample_data.x.size(1)
                final_gcn = self.gcn_model_class(
                    num_atom_features=num_atom_features,
                    **self.gcn_params
                ).to(self.device)
                
                optimizer = torch.optim.Adam(final_gcn.parameters(), lr=0.001)
                
                for epoch in range(self.gcn_epochs):
                    train_loss = train_one_epoch(final_gcn, full_loader, optimizer, self.device)
                    if epoch % 10 == 0:
                        logger.info(f"Final GCN Epoch {epoch + 1}: Loss {train_loss:.4f}")
                
                self.base_models_['gcn'] = final_gcn
            
        except Exception as e:
            logger.error(f"Error training final GCN model: {str(e)}")
        
        # Train final tree ensemble
        try:
            final_tree = TreeEnsemble(
                models=self.tree_models,
                optimize_hyperparams=self.optimize_tree_hyperparams,
                n_trials=self.tree_optuna_trials,
                random_state=self.random_state,
                **self.tree_params
            )
            final_tree.fit(X, y)
            self.base_models_['tree'] = final_tree
            
        except Exception as e:
            logger.error(f"Error training final tree ensemble: {str(e)}")
        
        logger.info("Stacking ensemble training completed!")
        return self
    
    def predict(self, df: pd.DataFrame, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted stacking ensemble.
        
        Args:
            df: DataFrame with SMILES for prediction
            X: Feature matrix for tree models
            
        Returns:
            Predictions array of shape (n_samples, n_targets)
        """
        if not self.meta_models_ and not self.base_models_:
            raise ValueError("Stacking ensemble must be fitted before making predictions")
        
        logger.info("Generating predictions with stacking ensemble...")
        
        n_samples = len(df)
        predictions = np.full((n_samples, self.n_targets_), np.nan)
        
        # Get base model predictions
        gcn_preds = np.full((n_samples, self.n_targets_), np.nan)
        tree_preds = np.full((n_samples, self.n_targets_), np.nan)
        
        # GCN predictions
        if 'gcn' in self.base_models_:
            try:
                from torch_geometric.data import Batch
                
                def collate_fn(batch):
                    valid_batch = [item for item in batch if item is not None]
                    if len(valid_batch) == 0:
                        return None
                    return Batch.from_data_list(valid_batch)
                
                test_dataset = PolymerDataset(df, target_cols=self.target_cols_, is_test=True)
                test_loader = DataLoader(
                    test_dataset, 
                    batch_size=self.batch_size, 
                    shuffle=False,
                    collate_fn=collate_fn
                )
                
                try:
                    _, gcn_preds = predict(self.base_models_['gcn'], test_loader, self.device)
                except Exception as e:
                    logger.error(f"Error in GCN prediction: {str(e)}")
                    gcn_preds = np.full((n_samples, self.n_targets_), np.nan)
                
            except Exception as e:
                logger.error(f"Error generating GCN predictions: {str(e)}")
        
        # Tree ensemble predictions
        if 'tree' in self.base_models_:
            try:
                tree_preds = self.base_models_['tree'].predict(X)
            except Exception as e:
                logger.error(f"Error generating tree predictions: {str(e)}")
        
        # Combine predictions using meta-models
        for target_idx, target_col in enumerate(self.target_cols_):
            if target_idx not in self.meta_models_:
                # Use simple average if no meta-model
                valid_gcn = ~np.isnan(gcn_preds[:, target_idx])
                valid_tree = ~np.isnan(tree_preds[:, target_idx])
                
                if np.any(valid_gcn) and np.any(valid_tree):
                    # Both models have predictions
                    both_valid = valid_gcn & valid_tree
                    predictions[both_valid, target_idx] = (
                        gcn_preds[both_valid, target_idx] + 
                        tree_preds[both_valid, target_idx]
                    ) / 2
                    
                    # Only GCN valid
                    only_gcn = valid_gcn & ~valid_tree
                    predictions[only_gcn, target_idx] = gcn_preds[only_gcn, target_idx]
                    
                    # Only tree valid
                    only_tree = ~valid_gcn & valid_tree
                    predictions[only_tree, target_idx] = tree_preds[only_tree, target_idx]
                
                elif np.any(valid_gcn):
                    predictions[valid_gcn, target_idx] = gcn_preds[valid_gcn, target_idx]
                elif np.any(valid_tree):
                    predictions[valid_tree, target_idx] = tree_preds[valid_tree, target_idx]
                
                continue
            
            # Use meta-model predictions
            meta_model = self.meta_models_[target_idx]
            
            # Prepare base predictions for meta-model
            base_pred_features = np.column_stack([
                gcn_preds[:, target_idx],
                tree_preds[:, target_idx]
            ])
            
            # Only predict where both base models have valid predictions
            valid_mask = ~np.isnan(base_pred_features).any(axis=1)
            
            if np.any(valid_mask):
                meta_preds = meta_model.predict(base_pred_features[valid_mask])
                predictions[valid_mask, target_idx] = meta_preds
        
        logger.info("Stacking ensemble predictions completed")
        return predictions
    
    def get_cv_scores(self) -> Dict[str, Dict[str, float]]:
        """Get cross-validation scores for base models."""
        return self.cv_scores_.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about fitted models."""
        info = {
            'n_targets': self.n_targets_,
            'target_cols': self.target_cols_,
            'cv_folds': self.cv_folds,
            'base_models': list(self.base_models_.keys()),
            'meta_models': list(self.meta_models_.keys()),
            'cv_scores': self.cv_scores_
        }
        
        if 'tree' in self.base_models_:
            info['tree_model_info'] = self.base_models_['tree'].get_model_info()
        
        return info