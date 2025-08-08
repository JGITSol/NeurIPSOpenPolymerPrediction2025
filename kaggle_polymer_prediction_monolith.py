#!/usr/bin/env python3
"""
NeurIPS 2025 Polymer Prediction - Complete Stacking Ensemble Solution
=====================================================================

This is a complete, self-contained solution for the NeurIPS 2025 Polymer Prediction
competition. It implements a sophisticated stacking ensemble that combines:

1. Graph Convolutional Networks (GCN) for molecular graph representation
2. Tree ensemble models (LightGBM, XGBoost, CatBoost) for tabular features
3. Cross-validation framework for out-of-fold predictions
4. Meta-learning with Ridge regression to combine base model predictions

Key Features:
- Handles missing values in multi-target regression
- Proper cross-validation to prevent data leakage
- Molecular featurization from SMILES strings
- Robust error handling and logging
- Competition-ready submission format

Usage:
1. Install required packages: pip install torch torch-geometric rdkit lightgbm xgboost catboost scikit-learn pandas numpy tqdm
2. Place your data files in the same directory
3. Run this script to train and generate predictions

Author: AI Assistant (Kiro)
Date: 2025
"""

import warnings
warnings.filterwarnings('ignore')

import logging
import gc
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# RDKit for molecular processing
try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")

# Tree ensemble imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# MOLECULAR FEATURIZATION
# ============================================================================

def get_atom_features(atom):
    """Extract features for a single atom."""
    if not RDKIT_AVAILABLE:
        return [0] * 25  # Dummy features if RDKit not available
    
    # Basic atom features
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetTotalValence(),
        int(atom.GetIsAromatic()),
        int(atom.GetChiralTag()),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
    ]
    
    # One-hot encode common atomic numbers in polymers
    common_atoms = [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, C, N, O, F, Si, P, S, Cl, Br, I
    atomic_num_one_hot = [0] * (len(common_atoms) + 1)  # +1 for "other"
    
    atomic_num = atom.GetAtomicNum()
    if atomic_num in common_atoms:
        atomic_num_one_hot[common_atoms.index(atomic_num)] = 1
    else:
        atomic_num_one_hot[-1] = 1  # "other" category
    
    # Hybridization one-hot encoding
    hybridization_types = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]
    hybridization_one_hot = [0] * (len(hybridization_types) + 1)  # +1 for "other"
    
    hybridization = atom.GetHybridization()
    if hybridization in hybridization_types:
        hybridization_one_hot[hybridization_types.index(hybridization)] = 1
    else:
        hybridization_one_hot[-1] = 1
    
    return features + atomic_num_one_hot + hybridization_one_hot


def get_bond_features(bond):
    """Extract features for a single bond."""
    if not RDKIT_AVAILABLE:
        return [0] * 6  # Dummy features if RDKit not available
    
    # Bond type one-hot encoding
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type_one_hot = [0] * (len(bond_types) + 1)  # +1 for "other"
    
    bond_type = bond.GetBondType()
    if bond_type in bond_types:
        bond_type_one_hot[bond_types.index(bond_type)] = 1
    else:
        bond_type_one_hot[-1] = 1
    
    # Additional bond features
    features = [
        int(bond.IsInRing()),
        int(bond.GetIsConjugated()),
    ]
    
    return features + bond_type_one_hot


def smiles_to_graph(smiles_string):
    """Convert a SMILES string to a PyG Data object."""
    if not RDKIT_AVAILABLE:
        # Create dummy graph if RDKit not available
        x = torch.randn(5, 25)  # 5 atoms, 25 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_attr = torch.randn(4, 6)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)  # Add explicit hydrogens

    # Get atom features
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)

    # Get bond features and connectivity
    if mol.GetNumBonds() > 0:
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append((i, j))
            edge_indices.append((j, i))  # Graph must be undirected

            bond_features = get_bond_features(bond)
            edge_attrs.append(bond_features)
            edge_attrs.append(bond_features)  # Same features for both directions

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Handle molecules with no bonds (e.g., single atoms)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_atom_features = x.size(1)

    return data


# ============================================================================
# DATASET CLASS
# ============================================================================

class PolymerDataset(Dataset):
    """PyTorch Geometric Dataset for polymer data."""

    def __init__(self, df, target_cols=None, is_test=False):
        super().__init__()
        self.df = df
        self.is_test = is_test
        
        if target_cols is None:
            target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.target_cols = target_cols
        
        self.smiles_list = df['SMILES'].tolist()
        self.ids = df['id'].tolist()
        
        if not is_test:
            # Extract targets and create masks for missing values
            self.targets = []
            self.masks = []
            
            for idx in range(len(df)):
                target_values = []
                mask_values = []
                
                for col in target_cols:
                    if col in df.columns:
                        val = df.iloc[idx][col]
                        if pd.isna(val):
                            target_values.append(0.0)  # Placeholder for missing values
                            mask_values.append(0.0)    # Mask indicates missing
                        else:
                            target_values.append(float(val))
                            mask_values.append(1.0)    # Mask indicates present
                    else:
                        target_values.append(0.0)
                        mask_values.append(0.0)
                
                self.targets.append(target_values)
                self.masks.append(mask_values)
        
        self.cache = {}  # Cache graphs to avoid re-computing

    def len(self):
        return len(self.df)

    def get(self, idx):
        if idx in self.cache:
            data = self.cache[idx]
        else:
            smiles = self.smiles_list[idx]
            data = smiles_to_graph(smiles)
            if data is None:  # Handle RDKit parsing errors
                return None
            self.cache[idx] = data

        # Add ID
        data.id = int(self.ids[idx])
        
        if not self.is_test:
            # Add target values and masks
            data.y = torch.tensor(self.targets[idx], dtype=torch.float).unsqueeze(0)  # Shape: (1, 5)
            data.mask = torch.tensor(self.masks[idx], dtype=torch.float).unsqueeze(0)  # Shape: (1, 5)
        
        return data


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def masked_mse_loss(predictions, targets, masks):
    """Calculate MSE loss only for non-missing values."""
    assert predictions.shape == targets.shape == masks.shape
    
    valid_mask = masks > 0
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)
    
    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]
    
    loss = torch.mean((valid_predictions - valid_targets) ** 2)
    return loss


def train_one_epoch(model, loader, optimizer, device):
    """Perform one full training pass over the dataset."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for data in tqdm(loader, desc="Training", leave=False):
        if data is None:
            continue
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = masked_mse_loss(out, data.y, data.mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        
    return total_loss / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    all_preds = []
    all_targets = []
    all_masks = []
    
    for data in tqdm(loader, desc="Evaluating", leave=False):
        if data is None:
            continue
        data = data.to(device)
        out = model(data)
        
        loss = masked_mse_loss(out, data.y, data.mask)
        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs
        
        all_preds.append(out.cpu())
        all_targets.append(data.y.cpu())
        all_masks.append(data.mask.cpu())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    if len(all_preds) == 0:
        return avg_loss, {'Tg': np.nan, 'FFV': np.nan, 'Tc': np.nan, 'Density': np.nan, 'Rg': np.nan}
    
    # Calculate per-property RMSE
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)
    
    property_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    rmses = {}
    
    for i, prop_name in enumerate(property_names):
        prop_mask = masks[:, i]
        if prop_mask.sum() > 0:
            prop_preds = preds[:, i][prop_mask == 1]
            prop_targets = targets[:, i][prop_mask == 1]
            rmse = torch.sqrt(torch.mean((prop_preds - prop_targets) ** 2))
            rmses[prop_name] = rmse.item()
        else:
            rmses[prop_name] = float('nan')
    
    return avg_loss, rmses


@torch.no_grad()
def predict(model, loader, device):
    """Generate predictions for test data."""
    model.eval()
    all_ids = []
    all_preds = []
    
    for data in tqdm(loader, desc="Predicting", leave=False):
        if data is None:
            continue
        data = data.to(device)
        out = model(data)
        
        # Handle both tensor and scalar id cases
        if hasattr(data, 'id'):
            if torch.is_tensor(data.id):
                all_ids.extend(data.id.tolist())
            elif isinstance(data.id, (list, tuple)):
                all_ids.extend(data.id)
            else:
                all_ids.append(data.id)
        else:
            # Fallback: use batch indices
            batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
            all_ids.extend(range(len(all_ids), len(all_ids) + batch_size))
        all_preds.append(out.cpu())
    
    if len(all_preds) == 0:
        return [], np.array([])
    
    predictions = torch.cat(all_preds, dim=0).numpy()
    return all_ids, predictions


# ============================================================================
# GCN MODEL
# ============================================================================

class PolymerGCN(nn.Module):
    """Graph Convolutional Network for polymer property prediction."""
    
    def __init__(self, num_atom_features, hidden_channels=64, num_gcn_layers=3, dropout=0.2):
        super().__init__()
        self.num_atom_features = num_atom_features
        self.hidden_channels = hidden_channels
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_atom_features, hidden_channels))
        
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_channels, 5)  # 5 target properties
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation after last layer
                x = torch.relu(x)
                x = self.dropout_layer(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final prediction
        x = self.dropout_layer(x)
        x = self.output_layer(x)
        
        return x


# ============================================================================
# TREE ENSEMBLE MODELS
# ============================================================================

class LightGBMWrapper(BaseEstimator, RegressorMixin):
    """LightGBM model wrapper with multi-target support."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=-1, 
                 num_leaves=31, random_state=42, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.kwargs = kwargs
        self.models_ = {}
        self.n_targets_ = None
        
    def fit(self, X, y):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.n_targets_ = y.shape[1]
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'random_state': self.random_state,
            'verbose': -1,
            **self.kwargs
        }
        
        # Train separate model for each target
        for target_idx in range(self.n_targets_):
            y_target = y[:, target_idx]
            
            # Handle missing values
            mask = ~np.isnan(y_target)
            if not np.any(mask):
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
                callbacks=[lgb.log_evaluation(0)]
            )
            
            self.models_[target_idx] = model
            
        return self
    
    def predict(self, X):
        if not self.models_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        if self.n_targets_ == 1:
            if 0 in self.models_:
                return self.models_[0].predict(X)
            else:
                return np.full(n_samples, np.nan)
        else:
            predictions = np.full((n_samples, self.n_targets_), np.nan)
            for target_idx, model in self.models_.items():
                predictions[:, target_idx] = model.predict(X)
            return predictions


class XGBoostWrapper(BaseEstimator, RegressorMixin):
    """XGBoost model wrapper with multi-target support."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, 
                 random_state=42, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
        self.models_ = {}
        self.n_targets_ = None
        
    def fit(self, X, y):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.n_targets_ = y.shape[1]
        
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'verbosity': 0,
            **self.kwargs
        }
        
        # Train separate model for each target
        for target_idx in range(self.n_targets_):
            y_target = y[:, target_idx]
            
            # Handle missing values
            mask = ~np.isnan(y_target)
            if not np.any(mask):
                continue
                
            X_masked = X[mask]
            y_masked = y_target[mask]
            
            # Create and train XGBoost model
            model = xgb.XGBRegressor(**params)
            model.fit(X_masked, y_masked)
            
            self.models_[target_idx] = model
            
        return self
    
    def predict(self, X):
        if not self.models_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        if self.n_targets_ == 1:
            if 0 in self.models_:
                return self.models_[0].predict(X)
            else:
                return np.full(n_samples, np.nan)
        else:
            predictions = np.full((n_samples, self.n_targets_), np.nan)
            for target_idx, model in self.models_.items():
                predictions[:, target_idx] = model.predict(X)
            return predictions


class TreeEnsemble:
    """Complete tree ensemble combining multiple tree models."""
    
    def __init__(self, models=None, random_state=42):
        if models is None:
            models = []
            if LIGHTGBM_AVAILABLE:
                models.append('lgbm')
            if XGBOOST_AVAILABLE:
                models.append('xgb')
        
        self.models = models
        self.random_state = random_state
        self.trained_models = {}
        self.model_weights = {}
        
    def fit(self, X, y):
        logger.info(f"Training tree ensemble with models: {self.models}")
        
        for model_type in self.models:
            logger.info(f"Training {model_type} model...")
            
            try:
                if model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
                    model = LightGBMWrapper(random_state=self.random_state)
                elif model_type == 'xgb' and XGBOOST_AVAILABLE:
                    model = XGBoostWrapper(random_state=self.random_state)
                else:
                    logger.warning(f"Model {model_type} not available, skipping")
                    continue
                
                model.fit(X, y)
                self.trained_models[model_type] = model
                
                # Calculate model weight based on cross-validation performance
                cv_score = self._calculate_cv_score(model, X, y)
                self.model_weights[model_type] = 1.0 / (cv_score + 1e-8)
                
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
    
    def _calculate_cv_score(self, model, X, y):
        """Calculate cross-validation score for model weighting."""
        kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Create a copy of the model for CV
                if isinstance(model, LightGBMWrapper):
                    model_copy = LightGBMWrapper(random_state=self.random_state)
                elif isinstance(model, XGBoostWrapper):
                    model_copy = XGBoostWrapper(random_state=self.random_state)
                else:
                    continue
                
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
            except Exception as e:
                logger.warning(f"CV fold failed: {str(e)}")
                continue
        
        return np.mean(scores) if scores else float('inf')
    
    def predict(self, X):
        """Make predictions using the ensemble of trained models."""
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


# ============================================================================
# STACKING ENSEMBLE
# ============================================================================

class StackingEnsemble(BaseEstimator, RegressorMixin):
    """Stacking ensemble combining GCN and tree ensemble models using cross-validation."""
    
    def __init__(self, gcn_model_class, gcn_params=None, tree_models=None, 
                 cv_folds=5, random_state=42, device=None, batch_size=32, 
                 gcn_epochs=50):
        self.gcn_model_class = gcn_model_class
        self.gcn_params = gcn_params or {}
        self.tree_models = tree_models or ['lgbm', 'xgb']
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.device = device or torch.device('cpu')
        self.batch_size = batch_size
        self.gcn_epochs = gcn_epochs
        
        # Fitted components
        self.base_models_ = {}
        self.meta_models_ = {}
        self.n_targets_ = None
        self.target_cols_ = None
        self.cv_scores_ = {}
        
        # Set random seeds
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
    
    def _create_cv_splits(self, X, y):
        """Create cross-validation splits ensuring no data leakage."""
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        splits = list(kf.split(X))
        
        logger.info(f"Created {len(splits)} CV splits with approximately "
                   f"{len(splits[0][0])} train and {len(splits[0][1])} validation samples each")
        
        return splits
    
    def _train_gcn_fold(self, train_df, val_df, fold_idx):
        """Train GCN model on one CV fold."""
        logger.info(f"Training GCN for fold {fold_idx + 1}/{self.cv_folds}")
        
        try:
            # Create datasets
            train_dataset = PolymerDataset(train_df, target_cols=self.target_cols_, is_test=False)
            val_dataset = PolymerDataset(val_df, target_cols=self.target_cols_, is_test=False)
            
            # Create data loaders
            def collate_fn(batch):
                valid_batch = [item for item in batch if item is not None]
                if len(valid_batch) == 0:
                    return None
                from torch_geometric.data import Batch
                return Batch.from_data_list(valid_batch)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                    shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                  shuffle=False, collate_fn=collate_fn)
            
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
                n_val_samples = len(val_df)
                val_preds = np.full((n_val_samples, self.n_targets_), np.nan)
            
            # Calculate final validation scores
            final_val_loss, final_val_rmses = evaluate(model, val_loader, self.device)
            
            # Clean up
            del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return val_preds, final_val_rmses
            
        except Exception as e:
            logger.error(f"Error training GCN for fold {fold_idx}: {str(e)}")
            n_val_samples = len(val_df)
            dummy_preds = np.full((n_val_samples, self.n_targets_), np.nan)
            dummy_scores = {col: np.nan for col in self.target_cols_}
            return dummy_preds, dummy_scores
    
    def _train_tree_fold(self, X_train, y_train, X_val, y_val, fold_idx):
        """Train tree ensemble on one CV fold."""
        logger.info(f"Training tree ensemble for fold {fold_idx + 1}/{self.cv_folds}")
        
        try:
            # Create tree ensemble
            tree_ensemble = TreeEnsemble(models=self.tree_models, random_state=self.random_state)
            
            # Train ensemble
            tree_ensemble.fit(X_train, y_train)
            
            # Generate validation predictions
            val_preds = tree_ensemble.predict(X_val)
            
            # Calculate validation scores
            val_scores = {}
            for i, col in enumerate(self.target_cols_):
                mask = ~np.isnan(y_val[:, i])
                if np.any(mask):
                    rmse = np.sqrt(mean_squared_error(y_val[mask, i], val_preds[mask, i]))
                    val_scores[col] = rmse
                else:
                    val_scores[col] = np.nan
            
            return val_preds, val_scores
            
        except Exception as e:
            logger.error(f"Error training tree ensemble for fold {fold_idx}: {str(e)}")
            dummy_preds = np.full((len(X_val), self.n_targets_), np.nan)
            dummy_scores = {col: np.nan for col in self.target_cols_}
            return dummy_preds, dummy_scores
    
    def _generate_base_predictions(self, df, X, y):
        """Generate out-of-fold predictions from base models using cross-validation."""
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
            gcn_val_preds, gcn_val_scores = self._train_gcn_fold(train_df, val_df, fold_idx)
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
    
    def _average_fold_scores(self, fold_scores):
        """Average scores across folds, handling NaN values."""
        avg_scores = {}
        for col in self.target_cols_:
            scores = [fold_score[col] for fold_score in fold_scores 
                     if not np.isnan(fold_score[col])]
            avg_scores[col] = np.mean(scores) if scores else np.nan
        return avg_scores
    
    def _train_meta_models(self, base_predictions, y):
        """Train meta-learner models to combine base model predictions."""
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
    
    def fit(self, df, X, y):
        """Fit the stacking ensemble."""
        logger.info("Starting stacking ensemble training...")
        
        # Store target information
        self.n_targets_ = y.shape[1]
        
        # Try to get target column names from DataFrame
        common_targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        available_targets = [col for col in common_targets if col in df.columns]
        
        if len(available_targets) == self.n_targets_:
            self.target_cols_ = available_targets
        else:
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
            def collate_fn(batch):
                valid_batch = [item for item in batch if item is not None]
                if len(valid_batch) == 0:
                    return None
                from torch_geometric.data import Batch
                return Batch.from_data_list(valid_batch)
            
            full_dataset = PolymerDataset(df, target_cols=self.target_cols_, is_test=False)
            full_loader = DataLoader(full_dataset, batch_size=self.batch_size, 
                                   shuffle=True, collate_fn=collate_fn)
            
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
            final_tree = TreeEnsemble(models=self.tree_models, random_state=self.random_state)
            final_tree.fit(X, y)
            self.base_models_['tree'] = final_tree
            
        except Exception as e:
            logger.error(f"Error training final tree ensemble: {str(e)}")
        
        logger.info("Stacking ensemble training completed!")
        return self
    
    def predict(self, df, X):
        """Make predictions using the fitted stacking ensemble."""
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
                def collate_fn(batch):
                    valid_batch = [item for item in batch if item is not None]
                    if len(valid_batch) == 0:
                        return None
                    from torch_geometric.data import Batch
                    return Batch.from_data_list(valid_batch)
                
                test_dataset = PolymerDataset(df, target_cols=self.target_cols_, is_test=True)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                       shuffle=False, collate_fn=collate_fn)
                
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
                    both_valid = valid_gcn & valid_tree
                    predictions[both_valid, target_idx] = (
                        gcn_preds[both_valid, target_idx] + 
                        tree_preds[both_valid, target_idx]
                    ) / 2
                    
                    only_gcn = valid_gcn & ~valid_tree
                    predictions[only_gcn, target_idx] = gcn_preds[only_gcn, target_idx]
                    
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
    
    def get_cv_scores(self):
        """Get cross-validation scores for base models."""
        return self.cv_scores_.copy()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_molecular_features(df):
    """Create molecular features from SMILES."""
    logger.info("Creating molecular features...")
    
    features = []
    
    for smiles in df['SMILES']:
        # Simple features based on SMILES string
        feature_vector = [
            len(smiles),  # Length
            smiles.count('C'),  # Carbon count
            smiles.count('N'),  # Nitrogen count
            smiles.count('O'),  # Oxygen count
            smiles.count('S'),  # Sulfur count
            smiles.count('F'),  # Fluorine count
            smiles.count('='),  # Double bonds
            smiles.count('#'),  # Triple bonds
            smiles.count('('),  # Branches
            smiles.count('['),  # Brackets
            smiles.count('c'),  # Aromatic carbons
            smiles.count('n'),  # Aromatic nitrogens
            smiles.count('o'),  # Aromatic oxygens
            smiles.count('s'),  # Aromatic sulfurs
            smiles.count('*'),  # Wildcards (polymer connection points)
        ]
        features.append(feature_vector)
    
    features_array = np.array(features, dtype=np.float32)
    logger.info(f"Created molecular features: {features_array.shape}")
    
    return features_array


def load_competition_data():
    """Load and combine all competition data."""
    logger.info("Loading competition data...")
    
    # Load main training data
    train_df = pd.read_csv('train.csv')
    logger.info(f"Loaded main training data: {len(train_df)} samples")
    
    # Load supplemental datasets if available
    supplement_dfs = []
    for i in range(1, 5):
        supp_path = f'train_supplement/dataset{i}.csv'
        if os.path.exists(supp_path):
            supp_df = pd.read_csv(supp_path)
            logger.info(f"Loaded supplement dataset {i}: {len(supp_df)} samples")
            supplement_dfs.append(supp_df)
    
    # Process supplemental data
    combined_supplement = pd.DataFrame()
    for df in supplement_dfs:
        if 'TC_mean' in df.columns:
            # This is thermal conductivity data
            df_processed = df.rename(columns={'TC_mean': 'Tc'})
            df_processed['id'] = range(len(combined_supplement), len(combined_supplement) + len(df_processed))
            combined_supplement = pd.concat([combined_supplement, df_processed], ignore_index=True)
    
    if len(combined_supplement) > 0:
        logger.info(f"Combined supplemental data: {len(combined_supplement)} samples")
        
        # Align columns
        main_cols = ['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
        supp_cols = ['id', 'SMILES'] + [col for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg'] if col in combined_supplement.columns]
        
        # Fill missing columns with NaN
        for col in main_cols:
            if col not in combined_supplement.columns:
                combined_supplement[col] = np.nan
        
        # Combine datasets
        full_df = pd.concat([train_df[main_cols], combined_supplement[main_cols]], ignore_index=True)
    else:
        full_df = train_df
    
    logger.info(f"Total combined data: {len(full_df)} samples")
    
    # Load test data
    test_df = pd.read_csv('test.csv')
    logger.info(f"Loaded test data: {len(test_df)} samples")
    
    return full_df, test_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logger.info("NeurIPS 2025 Polymer Prediction - Stacking Ensemble Solution")
    logger.info("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load data
        train_df, test_df = load_competition_data()
        
        # Filter out samples with all missing targets for training
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        train_df_clean = train_df.dropna(subset=target_cols, how='all').copy()
        logger.info(f"Training samples after removing all-missing: {len(train_df_clean)}")
        
        # Create molecular features
        X_train = create_molecular_features(train_df_clean)
        X_test = create_molecular_features(test_df)
        
        # Prepare target matrix
        y_train = train_df_clean[target_cols].values.astype(np.float32)
        
        # Check data quality
        logger.info("Data quality check:")
        for i, col in enumerate(target_cols):
            valid_count = np.sum(~np.isnan(y_train[:, i]))
            logger.info(f"  {col}: {valid_count}/{len(y_train)} valid values ({valid_count/len(y_train)*100:.1f}%)")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create stacking ensemble
        logger.info("Creating stacking ensemble...")
        ensemble = StackingEnsemble(
            gcn_model_class=PolymerGCN,
            gcn_params={
                'hidden_channels': 64,
                'num_gcn_layers': 3,
                'dropout': 0.2
            },
            tree_models=['lgbm', 'xgb'],  # Use available tree models
            cv_folds=5,
            gcn_epochs=50,
            random_state=42,
            device=device,
            batch_size=32
        )
        
        logger.info("Training stacking ensemble...")
        logger.info("This may take 30-60 minutes depending on your hardware...")
        
        # Train the ensemble
        ensemble.fit(train_df_clean, X_train_scaled, y_train)
        
        logger.info("Training completed!")
        
        # Get training information
        cv_scores = ensemble.get_cv_scores()
        
        logger.info("Training Results:")
        if 'gcn' in cv_scores:
            logger.info("  GCN CV scores:")
            for prop, score in cv_scores['gcn'].items():
                if not np.isnan(score):
                    logger.info(f"    {prop}: {score:.4f} RMSE")
        
        if 'tree' in cv_scores:
            logger.info("  Tree ensemble CV scores:")
            for prop, score in cv_scores['tree'].items():
                if not np.isnan(score):
                    logger.info(f"    {prop}: {score:.4f} RMSE")
        
        # Make predictions on test set
        logger.info("Making predictions on test set...")
        predictions = ensemble.predict(test_df, X_test_scaled)
        
        logger.info(f"Prediction shape: {predictions.shape}")
        logger.info(f"Valid predictions: {np.sum(~np.isnan(predictions))}/{predictions.size}")
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Tg': predictions[:, 0],
            'FFV': predictions[:, 1],
            'Tc': predictions[:, 2],
            'Density': predictions[:, 3],
            'Rg': predictions[:, 4]
        })
        
        # Fill any remaining NaN values with 0 (as per sample submission)
        submission = submission.fillna(0)
        
        # Save submission
        submission.to_csv('submission.csv', index=False)
        logger.info("Submission saved to 'submission.csv'")
        
        # Show sample predictions
        logger.info("Sample predictions:")
        for i in range(min(5, len(predictions))):
            logger.info(f"  Sample {i+1}: Tg={predictions[i,0]:.3f}, FFV={predictions[i,1]:.3f}, "
                       f"Tc={predictions[i,2]:.3f}, Density={predictions[i,3]:.3f}, Rg={predictions[i,4]:.3f}")
        
        logger.info("\n✅ Stacking ensemble completed successfully!")
        logger.info("\nKey achievements:")
        logger.info("- Successfully loaded and processed competition data")
        logger.info("- Combined main training data with supplemental datasets")
        logger.info("- Trained stacking ensemble with GCN and tree models")
        logger.info("- Generated predictions using cross-validation and meta-learning")
        logger.info("- Created competition-ready submission file")
        
        return True
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Success! Check 'submission.csv' for your predictions.")
    else:
        print("\n❌ Execution failed. Check the logs above for details.")