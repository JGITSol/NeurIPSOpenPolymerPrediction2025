#!/usr/bin/env python3
"""
Comprehensive Polymer Prediction Monolith for Kaggle Competition

This monolith file contains all necessary components for the NeurIPS Open Polymer 
Prediction 2025 challenge, including:
- Advanced molecular featurization with RDKit descriptors (200+ features)
- Polymer-specific Graph Neural Networks with repeat unit awareness
- Multi-task learning for joint property prediction
- Tree ensemble models (LightGBM, XGBoost, CatBoost)
- Stacking ensemble for optimal performance
- Robust error handling and memory management

Usage:
    python kaggle_polymer_prediction_monolith.py

Data paths:
    - Training data: /kaggle/input/neurips-open-polymer-prediction-2025/train.csv
    - Test data: /kaggle/input/neurips-open-polymer-prediction-2025/test.csv
    - Output: submission.csv
"""

import os
import sys
import warnings
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import gc
import time
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress RDKit warnings and errors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')

# Additional RDKit error suppression
import sys
from io import StringIO

class SuppressRDKitOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self._original_stderr

# Core scientific computing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
import joblib

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Graph neural networks
import torch_geometric
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm

# Chemistry and molecular features
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, rdchem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcTPSA
from rdkit.Chem.Descriptors import MolWt, MolLogP, NumHDonors, NumHAcceptors

# Tree ensemble models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available")

# Progress tracking
from tqdm import tqdm

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Configuration
class Config:
    """Global configuration for the polymer prediction pipeline."""
    
    # Data paths (Kaggle default)
    DATA_PATH = "/kaggle/input/neurips-open-polymer-prediction-2025"
    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"
    SUBMISSION_FILE = "submission.csv"
    
    # Target columns
    TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Model parameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64 if torch.cuda.is_available() else 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    PATIENCE = 15
    
    # GNN parameters
    HIDDEN_CHANNELS = 256
    NUM_GCN_LAYERS = 4
    DROPOUT = 0.3
    
    # Cross-validation
    N_FOLDS = 5
    
    # Ensemble parameters
    USE_STACKING = True
    OPTIMIZE_HYPERPARAMS = True
    N_TRIALS = 50
    
    # Memory management
    MAX_MEMORY_GB = 12
    ENABLE_CHECKPOINTING = True

config = Config()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MOLECULAR FEATURIZATION WITH 200+ RDKIT DESCRIPTORS
# ============================================================================

class MolecularFeaturizer:
    """Advanced molecular featurization with 200+ RDKit descriptors."""
    
    def __init__(self):
        """Initialize the featurizer with comprehensive descriptor sets."""
        self.descriptor_names = []
        self.scaler = StandardScaler()
        self.fitted = False
        
        # Get all available RDKit descriptors
        self.rdkit_descriptors = [
            (name, func) for name, func in Descriptors.descList
        ]
        
        logger.info(f"Initialized featurizer with {len(self.rdkit_descriptors)} RDKit descriptors")
    
    def get_atom_features(self, atom):
        """Extract comprehensive atom features."""
        features = []
        
        # Basic properties
        features.extend([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetTotalValence(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            int(atom.IsInRingSize(3)),
            int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)),
            int(atom.IsInRingSize(6)),
            int(atom.IsInRingSize(7)),
            int(atom.IsInRingSize(8)),
        ])
        
        # Chirality
        chiral_tag = atom.GetChiralTag()
        chiral_features = [0, 0, 0, 0]  # CHI_UNSPECIFIED, CHI_TETRAHEDRAL_CW, CHI_TETRAHEDRAL_CCW, CHI_OTHER
        if chiral_tag < len(chiral_features):
            chiral_features[chiral_tag] = 1
        features.extend(chiral_features)
        
        # Hybridization
        hybridization = atom.GetHybridization()
        hybrid_features = [0] * 8  # SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED, OTHER
        hybrid_map = {
            rdchem.HybridizationType.SP: 0,
            rdchem.HybridizationType.SP2: 1,
            rdchem.HybridizationType.SP3: 2,
            rdchem.HybridizationType.SP3D: 3,
            rdchem.HybridizationType.SP3D2: 4,
            rdchem.HybridizationType.UNSPECIFIED: 5,
            rdchem.HybridizationType.OTHER: 6,
        }
        if hybridization in hybrid_map:
            hybrid_features[hybrid_map[hybridization]] = 1
        else:
            hybrid_features[7] = 1  # Unknown
        features.extend(hybrid_features)
        
        # Atomic number one-hot (common elements in polymers)
        common_atoms = [1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, C, N, O, F, Si, P, S, Cl, Br, I
        atom_features = [0] * (len(common_atoms) + 1)
        atomic_num = atom.GetAtomicNum()
        if atomic_num in common_atoms:
            atom_features[common_atoms.index(atomic_num)] = 1
        else:
            atom_features[-1] = 1  # Other
        features.extend(atom_features)
        
        return features
    
    def get_bond_features(self, bond):
        """Extract comprehensive bond features."""
        features = []
        
        # Bond type
        bond_type = bond.GetBondType()
        bond_features = [0] * 5  # SINGLE, DOUBLE, TRIPLE, AROMATIC, OTHER
        bond_map = {
            rdchem.BondType.SINGLE: 0,
            rdchem.BondType.DOUBLE: 1,
            rdchem.BondType.TRIPLE: 2,
            rdchem.BondType.AROMATIC: 3,
        }
        if bond_type in bond_map:
            bond_features[bond_map[bond_type]] = 1
        else:
            bond_features[4] = 1  # Other
        features.extend(bond_features)
        
        # Additional bond properties
        features.extend([
            int(bond.IsInRing()),
            int(bond.GetIsConjugated()),
            int(bond.GetStereo()),
        ])
        
        return features
    
    def smiles_to_graph(self, smiles):
        """Convert SMILES to graph with comprehensive features."""
        try:
            with SuppressRDKitOutput():
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
            
            mol = Chem.AddHs(mol)
            
            # Atom features
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append(self.get_atom_features(atom))
            
            if not atom_features:
                return None
            
            x = torch.tensor(atom_features, dtype=torch.float)
            
            # Bond features and connectivity
            edge_indices = []
            edge_attrs = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                bond_features = self.get_bond_features(bond)
                
                edge_indices.extend([(i, j), (j, i)])
                edge_attrs.extend([bond_features, bond_features])
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(self.get_bond_features(mol.GetBonds()[0])) if mol.GetNumBonds() > 0 else 8), dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data.num_atom_features = x.size(1)
            
            return data
            
        except Exception as e:
            logger.warning(f"Error converting SMILES to graph: {e}")
            return None
    
    def get_rdkit_descriptors(self, smiles):
        """Extract all available RDKit molecular descriptors."""
        try:
            with SuppressRDKitOutput():
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return [0.0] * len(self.rdkit_descriptors)
            
            descriptors = []
            for name, func in self.rdkit_descriptors:
                try:
                    value = func(mol)
                    if isinstance(value, (int, float)):
                        if np.isnan(value) or np.isinf(value):
                            descriptors.append(0.0)
                        else:
                            descriptors.append(float(value))
                    else:
                        descriptors.append(0.0)
                except Exception:
                    # Silently handle descriptor calculation failures
                    descriptors.append(0.0)
            
            return descriptors
            
        except Exception:
            # Return zeros for completely failed molecules
            return [0.0] * len(self.rdkit_descriptors)
    
    def get_polymer_specific_features(self, smiles):
        """Extract polymer-specific features without using SMARTS patterns."""
        try:
            with SuppressRDKitOutput():
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return [0.0] * 20  # Return default features
            
            features = []
            
            # Count atoms by element (safer than SMARTS)
            atom_counts = {}
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
            
            # Common polymer elements
            polymer_elements = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'Si']
            for element in polymer_elements:
                features.append(atom_counts.get(element, 0))
            
            # Bond type counts (safer approach)
            bond_counts = {'SINGLE': 0, 'DOUBLE': 0, 'TRIPLE': 0, 'AROMATIC': 0}
            for bond in mol.GetBonds():
                bond_type = str(bond.GetBondType())
                if bond_type in bond_counts:
                    bond_counts[bond_type] += 1
            
            features.extend([bond_counts['SINGLE'], bond_counts['DOUBLE'], 
                           bond_counts['TRIPLE'], bond_counts['AROMATIC']])
            
            # Ring information
            ring_info = mol.GetRingInfo()
            features.extend([
                ring_info.NumRings(),
                len([r for r in ring_info.AtomRings() if len(r) == 5]),  # 5-membered rings
                len([r for r in ring_info.AtomRings() if len(r) == 6]),  # 6-membered rings
            ])
            
            # Basic molecular properties
            try:
                basic_props = [
                    mol.GetNumAtoms(),
                    mol.GetNumBonds(),
                    mol.GetNumHeavyAtoms(),
                ]
                features.extend(basic_props)
            except Exception:
                features.extend([0.0] * 3)
            
            # Pad to ensure we have exactly 20 features
            while len(features) < 20:
                features.append(0.0)
            
            return features[:20]  # Ensure exactly 20 features
            
        except Exception:
            # Return zeros for completely failed molecules
            return [0.0] * 20
    
    def featurize_molecules(self, smiles_list):
        """Featurize a list of SMILES strings."""
        logger.info(f"Featurizing {len(smiles_list)} molecules...")
        
        all_features = []
        valid_indices = []
        
        for i, smiles in enumerate(tqdm(smiles_list, desc="Featurizing")):
            # RDKit descriptors
            rdkit_features = self.get_rdkit_descriptors(smiles)
            
            # Polymer-specific features
            polymer_features = self.get_polymer_specific_features(smiles)
            
            # Combine all features
            combined_features = rdkit_features + polymer_features
            
            # Check for valid features
            if not any(np.isnan(combined_features)) and not any(np.isinf(combined_features)):
                all_features.append(combined_features)
                valid_indices.append(i)
            else:
                # Replace invalid values
                combined_features = [0.0 if (np.isnan(x) or np.isinf(x)) else x for x in combined_features]
                all_features.append(combined_features)
                valid_indices.append(i)
        
        features_array = np.array(all_features)
        
        # Store descriptor names for reference
        if not self.descriptor_names:
            self.descriptor_names = [name for name, _ in self.rdkit_descriptors]
            self.descriptor_names.extend([f"polymer_feature_{i}" for i in range(20)])
        
        logger.info(f"Generated {features_array.shape[1]} molecular features")
        return features_array, valid_indices
    
    def fit_scaler(self, features):
        """Fit the feature scaler."""
        self.scaler.fit(features)
        self.fitted = True
        logger.info("Feature scaler fitted")
    
    def transform_features(self, features):
        """Transform features using fitted scaler."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transforming")
        return self.scaler.transform(features)

# ============================================================================
# POLYMER-SPECIFIC GRAPH NEURAL NETWORK
# ============================================================================

class PolymerGNN(nn.Module):
    """
    Polymer-specific Graph Neural Network with repeat unit awareness
    and multi-task learning capabilities.
    """
    
    def __init__(
        self,
        num_atom_features,
        num_edge_features=8,
        hidden_channels=256,
        num_layers=4,
        num_targets=5,
        dropout=0.3,
        use_attention=True,
        use_residual=True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Input projection
        self.atom_encoder = nn.Linear(num_atom_features, hidden_channels)
        self.edge_encoder = nn.Linear(num_edge_features, hidden_channels) if num_edge_features > 0 else None
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                conv = GATConv(
                    hidden_channels, 
                    hidden_channels // 8,  # 8 attention heads
                    heads=8,
                    dropout=dropout,
                    edge_dim=hidden_channels if self.edge_encoder else None
                )
            else:
                conv = GCNConv(hidden_channels, hidden_channels)
            
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Polymer-specific layers
        self.polymer_attention = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Global pooling
        self.global_pool = global_mean_pool
        
        # Multi-task prediction heads
        self.dropout = nn.Dropout(dropout)
        
        # Shared representation
        self.shared_layers = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels // 2, hidden_channels // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 4, 1)
            ) for _ in range(num_targets)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, data):
        """Forward pass with polymer-specific processing."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        
        # Encode features
        x = self.atom_encoder(x)
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # Graph convolutions with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_residual = x
            
            if self.use_attention and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection
            if self.use_residual and i > 0:
                x = x + x_residual
        
        # Polymer-specific attention mechanism
        # Group nodes by molecule for attention
        batch_size = batch.max().item() + 1
        pooled_representations = []
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                mol_x = x[mask].unsqueeze(0)  # (1, num_atoms, hidden_dim)
                
                # Self-attention for polymer repeat unit awareness
                attended_x, _ = self.polymer_attention(mol_x, mol_x, mol_x)
                
                # Global pooling
                pooled = attended_x.mean(dim=1)  # (1, hidden_dim)
                pooled_representations.append(pooled)
        
        if pooled_representations:
            x_global = torch.cat(pooled_representations, dim=0)
        else:
            # Fallback to standard global pooling
            x_global = self.global_pool(x, batch)
        
        # Shared representation
        shared_repr = self.shared_layers(x_global)
        
        # Multi-task predictions
        predictions = []
        for head in self.task_heads:
            pred = head(shared_repr)
            predictions.append(pred)
        
        return torch.cat(predictions, dim=1)

# ============================================================================
# DATASET AND DATA LOADING
# ============================================================================

class PolymerDataset(torch_geometric.data.Dataset):
    """Dataset for polymer graphs with multi-target support."""
    
    def __init__(self, df, featurizer, target_cols=None, is_test=False):
        super().__init__()
        self.df = df
        self.featurizer = featurizer
        self.is_test = is_test
        self.target_cols = target_cols or config.TARGET_COLS
        
        self.smiles_list = df['SMILES'].tolist()
        self.ids = df['id'].tolist()
        
        if not is_test:
            self.targets = []
            self.masks = []
            
            for idx in range(len(df)):
                target_values = []
                mask_values = []
                
                for col in self.target_cols:
                    if col in df.columns:
                        val = df.iloc[idx][col]
                        if pd.isna(val):
                            target_values.append(0.0)
                            mask_values.append(0.0)
                        else:
                            target_values.append(float(val))
                            mask_values.append(1.0)
                    else:
                        target_values.append(0.0)
                        mask_values.append(0.0)
                
                self.targets.append(target_values)
                self.masks.append(mask_values)
        
        self.cache = {}
    
    def len(self):
        return len(self.df)
    
    def get(self, idx):
        if idx in self.cache:
            data = self.cache[idx]
        else:
            smiles = self.smiles_list[idx]
            data = self.featurizer.smiles_to_graph(smiles)
            if data is None:
                return None
            self.cache[idx] = data
        
        data.id = int(self.ids[idx])
        
        if not self.is_test:
            data.y = torch.tensor(self.targets[idx], dtype=torch.float).unsqueeze(0)
            data.mask = torch.tensor(self.masks[idx], dtype=torch.float).unsqueeze(0)
        
        return data

def create_data_loaders(train_df, test_df, featurizer, batch_size=None):
    """Create data loaders for training and testing."""
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    train_dataset = PolymerDataset(train_df, featurizer, is_test=False)
    test_dataset = PolymerDataset(test_df, featurizer, is_test=True)
    
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return Batch.from_data_list(batch)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, test_loader

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def masked_mse_loss(predictions, targets, masks):
    """Compute MSE loss with missing value masks."""
    masked_predictions = predictions * masks
    masked_targets = targets * masks
    
    # Compute loss only where mask is 1
    loss = F.mse_loss(masked_predictions, masked_targets, reduction='none')
    loss = loss * masks
    
    # Average over valid entries
    valid_count = masks.sum()
    if valid_count > 0:
        return loss.sum() / valid_count
    else:
        return torch.tensor(0.0, device=predictions.device)

def weighted_mae_loss(predictions, targets, masks, weights=None):
    """Compute weighted MAE loss for competition metric."""
    if weights is None:
        weights = torch.ones(predictions.size(1), device=predictions.device)
    
    masked_predictions = predictions * masks
    masked_targets = targets * masks
    
    # Compute MAE for each target
    mae_per_target = torch.abs(masked_predictions - masked_targets) * masks
    
    # Weight by target importance
    weighted_mae = mae_per_target * weights.unsqueeze(0)
    
    # Average over valid entries
    valid_count = masks.sum(dim=0)
    target_mae = weighted_mae.sum(dim=0) / (valid_count + 1e-8)
    
    return target_mae.mean()

def train_epoch(model, loader, optimizer, device, criterion=masked_mse_loss):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        if batch is None:
            continue
        
        # Check if batch has required attributes
        if not (hasattr(batch, 'y') and hasattr(batch, 'mask')):
            logger.warning("Batch missing targets or masks in training")
            continue
        
        batch = batch.to(device)
        optimizer.zero_grad()
        
        predictions = model(batch)
        loss = criterion(predictions, batch.y, batch.mask)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs
    
    return total_loss / max(total_samples, 1)

@torch.no_grad()
def evaluate_model(model, loader, device, criterion=masked_mse_loss):
    """Evaluate model performance."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    all_masks = []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        if batch is None:
            continue
        
        # Check if batch has targets and masks
        if not (hasattr(batch, 'y') and hasattr(batch, 'mask')):
            logger.warning("Batch missing targets or masks in evaluation")
            continue
        
        batch = batch.to(device)
        predictions = model(batch)
        loss = criterion(predictions, batch.y, batch.mask)
        
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs
        
        all_predictions.append(predictions.cpu())
        all_targets.append(batch.y.cpu())
        all_masks.append(batch.mask.cpu())
    
    avg_loss = total_loss / max(total_samples, 1)
    
    if all_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        masks = torch.cat(all_masks, dim=0)
        
        # Calculate per-target metrics
        target_metrics = {}
        for i, col in enumerate(config.TARGET_COLS):
            mask = masks[:, i].bool()
            if mask.sum() > 0:
                pred_vals = predictions[mask, i]
                true_vals = targets[mask, i]
                
                mae = F.l1_loss(pred_vals, true_vals).item()
                mse = F.mse_loss(pred_vals, true_vals).item()
                
                target_metrics[col] = {'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse)}
        
        return avg_loss, target_metrics
    
    return avg_loss, {}

@torch.no_grad()
def predict_with_model(model, loader, device):
    """Generate predictions using trained model."""
    model.eval()
    all_predictions = []
    all_ids = []
    
    for batch in tqdm(loader, desc="Predicting", leave=False):
        if batch is None:
            continue
        
        batch = batch.to(device)
        predictions = model(batch)
        
        all_predictions.append(predictions.cpu())
        all_ids.extend([int(id_val) for id_val in batch.id])
    
    if all_predictions:
        predictions = torch.cat(all_predictions, dim=0).numpy()
        return all_ids, predictions
    
    return [], np.array([])

# ============================================================================
# TREE ENSEMBLE MODELS
# ============================================================================

class TreeEnsembleModel:
    """Ensemble of tree-based models for polymer prediction."""
    
    def __init__(self, models=None, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.fitted = False
        
        # Initialize available models
        if models is None:
            models = []
            if LIGHTGBM_AVAILABLE:
                models.append('lgbm')
            if XGBOOST_AVAILABLE:
                models.append('xgb')
            if CATBOOST_AVAILABLE:
                models.append('catboost')
        
        self.model_types = models
        logger.info(f"Initialized tree ensemble with models: {models}")
    
    def _create_lgbm_model(self, **params):
        """Create LightGBM model."""
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        }
        default_params.update(params)
        return default_params
    
    def _create_xgb_model(self, **params):
        """Create XGBoost model."""
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'verbosity': 0
        }
        default_params.update(params)
        return xgb.XGBRegressor(**default_params)
    
    def _create_catboost_model(self, **params):
        """Create CatBoost model."""
        default_params = {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'depth': 6,
            'learning_rate': 0.1,
            'iterations': 100,
            'random_state': self.random_state,
            'verbose': False
        }
        default_params.update(params)
        return cb.CatBoostRegressor(**default_params)
    
    def fit(self, X, y):
        """Fit all tree models."""
        logger.info("Training tree ensemble models...")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_targets = y.shape[1]
        
        for model_type in self.model_types:
            logger.info(f"Training {model_type} models...")
            self.models[model_type] = {}
            
            for target_idx in range(n_targets):
                y_target = y[:, target_idx]
                mask = ~np.isnan(y_target)
                
                if not np.any(mask):
                    logger.warning(f"No valid targets for {model_type} target {target_idx}")
                    continue
                
                X_masked = X[mask]
                y_masked = y_target[mask]
                
                try:
                    if model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
                        params = self._create_lgbm_model()
                        train_data = lgb.Dataset(X_masked, label=y_masked)
                        model = lgb.train(params, train_data, num_boost_round=100, verbose_eval=False)
                    
                    elif model_type == 'xgb' and XGBOOST_AVAILABLE:
                        model = self._create_xgb_model(n_estimators=100)
                        model.fit(X_masked, y_masked)
                    
                    elif model_type == 'catboost' and CATBOOST_AVAILABLE:
                        model = self._create_catboost_model(iterations=100)
                        model.fit(X_masked, y_masked)
                    
                    else:
                        continue
                    
                    self.models[model_type][target_idx] = model
                    
                except Exception as e:
                    logger.error(f"Error training {model_type} for target {target_idx}: {e}")
                    continue
        
        self.fitted = True
        logger.info("Tree ensemble training completed")
    
    def predict(self, X):
        """Generate predictions from all models."""
        if not self.fitted:
            raise ValueError("Models must be fitted before prediction")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_targets = len(config.TARGET_COLS)
        
        # Collect predictions from all models
        all_predictions = {}
        
        for model_type, models in self.models.items():
            predictions = np.full((n_samples, n_targets), np.nan)
            
            for target_idx, model in models.items():
                try:
                    if model_type == 'lgbm':
                        pred = model.predict(X)
                    else:
                        pred = model.predict(X)
                    
                    predictions[:, target_idx] = pred
                    
                except Exception as e:
                    logger.warning(f"Error predicting with {model_type} target {target_idx}: {e}")
                    continue
            
            all_predictions[model_type] = predictions
        
        return all_predictions

# ============================================================================
# STACKING ENSEMBLE
# ============================================================================

class StackingEnsemble:
    """Stacking ensemble combining GNN and tree models."""
    
    def __init__(self, meta_model=None, cv_folds=5, random_state=42):
        self.meta_model = meta_model or Ridge(alpha=1.0, random_state=random_state)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.base_models = {}
        self.fitted = False
    
    def add_base_model(self, name, model):
        """Add a base model to the ensemble."""
        self.base_models[name] = model
    
    def fit(self, X_tabular, X_graph_loader, y):
        """Fit the stacking ensemble."""
        logger.info("Training stacking ensemble...")
        
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_samples, n_targets = y.shape
        
        # Generate out-of-fold predictions for meta-model training
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Placeholder for meta-features
        meta_features = []
        meta_targets = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_tabular)):
            logger.info(f"Processing fold {fold + 1}/{self.cv_folds}")
            
            fold_predictions = []
            
            # Tree models on tabular features
            if 'tree_ensemble' in self.base_models:
                X_train_fold = X_tabular[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X_tabular[val_idx]
                
                tree_model = clone(self.base_models['tree_ensemble'])
                tree_model.fit(X_train_fold, y_train_fold)
                tree_preds = tree_model.predict(X_val_fold)
                
                # Average predictions across tree models
                if isinstance(tree_preds, dict):
                    tree_pred_avg = np.mean([pred for pred in tree_preds.values()], axis=0)
                else:
                    tree_pred_avg = tree_preds
                
                fold_predictions.append(tree_pred_avg)
            
            # Add GNN predictions if available
            # (This would require more complex cross-validation for graph data)
            
            # Combine fold predictions
            if fold_predictions:
                combined_preds = np.concatenate(fold_predictions, axis=1)
                meta_features.append(combined_preds)
                meta_targets.append(y[val_idx])
        
        if meta_features:
            # Train meta-model
            X_meta = np.vstack(meta_features)
            y_meta = np.vstack(meta_targets)
            
            # Train separate meta-model for each target
            self.meta_models = {}
            for target_idx in range(n_targets):
                y_target = y_meta[:, target_idx]
                mask = ~np.isnan(y_target)
                
                if np.any(mask):
                    meta_model = clone(self.meta_model)
                    meta_model.fit(X_meta[mask], y_target[mask])
                    self.meta_models[target_idx] = meta_model
        
        # Train final base models on full data
        for name, model in self.base_models.items():
            if name == 'tree_ensemble':
                model.fit(X_tabular, y)
        
        self.fitted = True
        logger.info("Stacking ensemble training completed")
    
    def predict(self, X_tabular, X_graph_loader=None):
        """Generate ensemble predictions."""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        base_predictions = []
        
        # Get tree model predictions
        if 'tree_ensemble' in self.base_models:
            tree_preds = self.base_models['tree_ensemble'].predict(X_tabular)
            if isinstance(tree_preds, dict):
                tree_pred_avg = np.mean([pred for pred in tree_preds.values()], axis=0)
            else:
                tree_pred_avg = tree_preds
            base_predictions.append(tree_pred_avg)
        
        # Combine base predictions
        if base_predictions:
            X_meta = np.concatenate(base_predictions, axis=1)
            
            # Generate meta-predictions
            n_samples = X_meta.shape[0]
            n_targets = len(config.TARGET_COLS)
            final_predictions = np.zeros((n_samples, n_targets))
            
            for target_idx in range(n_targets):
                if target_idx in self.meta_models:
                    final_predictions[:, target_idx] = self.meta_models[target_idx].predict(X_meta)
                else:
                    # Fallback to average of base predictions
                    final_predictions[:, target_idx] = X_meta[:, target_idx] if X_meta.shape[1] > target_idx else 0
            
            return final_predictions
        
        return np.zeros((X_tabular.shape[0], len(config.TARGET_COLS)))

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def load_data():
    """Load training and test data."""
    logger.info("Loading data...")
    
    # Try Kaggle paths first, then local paths
    train_paths = [
        os.path.join(config.DATA_PATH, config.TRAIN_FILE),
        "info/train.csv",
        "train.csv"
    ]
    
    test_paths = [
        os.path.join(config.DATA_PATH, config.TEST_FILE),
        "info/test.csv", 
        "test.csv"
    ]
    
    train_df = None
    test_df = None
    
    for path in train_paths:
        if os.path.exists(path):
            train_df = pd.read_csv(path)
            logger.info(f"Loaded training data from {path}: {train_df.shape}")
            break
    
    for path in test_paths:
        if os.path.exists(path):
            test_df = pd.read_csv(path)
            logger.info(f"Loaded test data from {path}: {test_df.shape}")
            break
    
    if train_df is None or test_df is None:
        raise FileNotFoundError("Could not find training or test data files")
    
    # Validate data
    required_train_cols = ['id', 'SMILES'] + config.TARGET_COLS
    required_test_cols = ['id', 'SMILES']
    
    missing_train_cols = set(required_train_cols) - set(train_df.columns)
    missing_test_cols = set(required_test_cols) - set(test_df.columns)
    
    if missing_train_cols:
        raise ValueError(f"Missing columns in training data: {missing_train_cols}")
    if missing_test_cols:
        raise ValueError(f"Missing columns in test data: {missing_test_cols}")
    
    # Remove invalid SMILES
    def is_valid_smiles(smiles):
        try:
            if not isinstance(smiles, str) or len(smiles.strip()) == 0:
                return False
            with SuppressRDKitOutput():
                mol = Chem.MolFromSmiles(smiles.strip())
                if mol is None:
                    return False
                # Additional validation - ensure molecule has atoms
                if mol.GetNumAtoms() == 0:
                    return False
                return True
        except Exception:
            return False
    
    train_valid = train_df['SMILES'].apply(is_valid_smiles)
    test_valid = test_df['SMILES'].apply(is_valid_smiles)
    
    logger.info(f"Valid SMILES - Train: {train_valid.sum()}/{len(train_df)}, Test: {test_valid.sum()}/{len(test_df)}")
    
    train_df = train_df[train_valid].reset_index(drop=True)
    test_df = test_df[test_valid].reset_index(drop=True)
    
    return train_df, test_df

def train_gnn_model(train_loader, val_loader, device):
    """Train the Graph Neural Network model."""
    logger.info("Training GNN model...")
    
    # Get sample to determine feature dimensions
    sample_batch = next(iter(train_loader))
    if sample_batch is None:
        raise ValueError("No valid batches in training data")
    
    num_atom_features = sample_batch.x.size(1)
    num_edge_features = sample_batch.edge_attr.size(1) if hasattr(sample_batch, 'edge_attr') else 0
    
    logger.info(f"Atom features: {num_atom_features}, Edge features: {num_edge_features}")
    
    # Initialize model
    model = PolymerGNN(
        num_atom_features=num_atom_features,
        num_edge_features=num_edge_features,
        hidden_channels=config.HIDDEN_CHANNELS,
        num_layers=config.NUM_GCN_LAYERS,
        num_targets=len(config.TARGET_COLS),
        dropout=config.DROPOUT
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.NUM_EPOCHS):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, device)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if val_metrics:
                for target, metrics in val_metrics.items():
                    logger.info(f"  {target}: MAE = {metrics['mae']:.4f}, RMSE = {metrics['rmse']:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_gnn_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Memory cleanup
        if epoch % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Load best model
    model.load_state_dict(torch.load('best_gnn_model.pt'))
    logger.info("GNN training completed")
    
    return model

def main():
    """Main execution pipeline."""
    logger.info("Starting Polymer Prediction Pipeline")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Target columns: {config.TARGET_COLS}")
    
    try:
        # Load data
        train_df, test_df = load_data()
        
        # Initialize molecular featurizer
        featurizer = MolecularFeaturizer()
        
        # Generate molecular features for tree models
        logger.info("Generating molecular features for tree models...")
        train_features, train_valid_idx = featurizer.featurize_molecules(train_df['SMILES'].tolist())
        test_features, test_valid_idx = featurizer.featurize_molecules(test_df['SMILES'].tolist())
        
        # Fit and transform features
        featurizer.fit_scaler(train_features)
        train_features_scaled = featurizer.transform_features(train_features)
        test_features_scaled = featurizer.transform_features(test_features)
        
        # Prepare targets for tree models
        train_targets = train_df[config.TARGET_COLS].values
        
        # Create graph data loaders
        logger.info("Creating graph data loaders...")
        train_loader, test_loader = create_data_loaders(train_df, test_df, featurizer)
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        
        train_idx, val_idx = train_test_split(
            range(len(train_df)), 
            test_size=0.2, 
            random_state=RANDOM_SEED,
            stratify=None
        )
        
        train_subset = train_df.iloc[train_idx].reset_index(drop=True)
        val_subset = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Create validation dataset with targets (not as test data)
        train_dataset_subset = PolymerDataset(train_subset, featurizer, is_test=False)
        val_dataset = PolymerDataset(val_subset, featurizer, is_test=False)  # Keep targets for validation
        
        def collate_fn(batch):
            batch = [item for item in batch if item is not None]
            if len(batch) == 0:
                return None
            return Batch.from_data_list(batch)
        
        train_loader_subset = DataLoader(train_dataset_subset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        # Train models
        predictions = {}
        
        # 1. Train Tree Ensemble
        logger.info("Training tree ensemble models...")
        tree_ensemble = TreeEnsembleModel()
        tree_ensemble.fit(train_features_scaled, train_targets)
        tree_predictions = tree_ensemble.predict(test_features_scaled)
        
        # Average tree model predictions
        if tree_predictions:
            tree_pred_avg = np.mean([pred for pred in tree_predictions.values()], axis=0)
            predictions['tree_ensemble'] = tree_pred_avg
        
        # 2. Train GNN Model
        logger.info("Training Graph Neural Network...")
        try:
            gnn_model = train_gnn_model(train_loader_subset, val_loader, config.DEVICE)
            
            # Generate GNN predictions
            test_ids, gnn_predictions = predict_with_model(gnn_model, test_loader, config.DEVICE)
            predictions['gnn'] = gnn_predictions
            
        except Exception as e:
            logger.error(f"GNN training failed: {e}")
            logger.info("Continuing with tree ensemble only...")
        
        # 3. Ensemble predictions
        if len(predictions) > 1:
            logger.info("Combining predictions with ensemble...")
            # Simple averaging for now
            final_predictions = np.mean([pred for pred in predictions.values()], axis=0)
        else:
            final_predictions = list(predictions.values())[0]
        
        # Create submission
        logger.info("Creating submission file...")
        submission = pd.DataFrame({'id': test_df['id']})
        
        for i, col in enumerate(config.TARGET_COLS):
            if i < final_predictions.shape[1]:
                submission[col] = final_predictions[:, i]
            else:
                submission[col] = 0.0
        
        # Validate submission
        for col in config.TARGET_COLS:
            if submission[col].isna().any():
                logger.warning(f"Found NaN values in {col}, filling with 0")
                submission[col].fillna(0, inplace=True)
            
            if np.isinf(submission[col]).any():
                logger.warning(f"Found infinite values in {col}, clipping")
                submission[col] = np.clip(submission[col], -1e6, 1e6)
        
        # Save submission
        submission.to_csv(config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {config.SUBMISSION_FILE}")
        logger.info(f"Submission shape: {submission.shape}")
        logger.info(f"Sample predictions:\n{submission.head()}")
        
        # Print summary statistics
        logger.info("\nPrediction Summary:")
        for col in config.TARGET_COLS:
            values = submission[col]
            logger.info(f"{col}: mean={values.mean():.4f}, std={values.std():.4f}, min={values.min():.4f}, max={values.max():.4f}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create dummy submission as fallback
        logger.info("Creating dummy submission as fallback...")
        try:
            if 'test_df' in locals():
                dummy_submission = pd.DataFrame({'id': test_df['id']})
                for col in config.TARGET_COLS:
                    dummy_submission[col] = 0.0
                
                dummy_submission.to_csv('dummy_' + config.SUBMISSION_FILE, index=False)
                logger.info(f"Dummy submission saved to dummy_{config.SUBMISSION_FILE}")
        except:
            logger.error("Failed to create dummy submission")
        
        raise

if __name__ == "__main__":
    main()