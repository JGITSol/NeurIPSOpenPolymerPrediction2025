#!/usr/bin/env python3
"""
Minimal Polymer Prediction Pipeline for Kaggle Competition
Focuses on core functionality without problematic SMARTS patterns.
"""

import os
import sys
import warnings
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import gc
import time

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress RDKit warnings completely
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Core imports
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Graph neural networks
import torch_geometric
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

# Chemistry
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, rdchem

# Tree models (optional)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from tqdm import tqdm

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Data paths
DATA_PATH = "/kaggle/input/neurips-open-polymer-prediction-2025"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "submission.csv"

TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64 if torch.cuda.is_available() else 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
HIDDEN_CHANNELS = 256
NUM_GCN_LAYERS = 4

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFeaturizer:
    """Simple molecular featurizer without SMARTS patterns."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def get_atom_features(self, atom):
        """Extract basic atom features."""
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetTotalValence(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetFormalCharge(),
        ]
        
        # One-hot encode common atoms
        common_atoms = [1, 6, 7, 8, 9, 16, 17]  # H, C, N, O, F, S, Cl
        atom_one_hot = [0] * (len(common_atoms) + 1)
        atomic_num = atom.GetAtomicNum()
        if atomic_num in common_atoms:
            atom_one_hot[common_atoms.index(atomic_num)] = 1
        else:
            atom_one_hot[-1] = 1
        
        return features + atom_one_hot
    
    def get_bond_features(self, bond):
        """Extract basic bond features."""
        bond_type = bond.GetBondType()
        features = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC),
            int(bond.IsInRing()),
            int(bond.GetIsConjugated()),
        ]
        return features
    
    def smiles_to_graph(self, smiles):
        """Convert SMILES to graph."""
        try:
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
                edge_attr = torch.empty((0, 6), dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            return data
            
        except Exception as e:
            logger.warning(f"Error converting SMILES to graph: {e}")
            return None
    
    def get_molecular_descriptors(self, smiles):
        """Get basic molecular descriptors."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [0.0] * 10
            
            descriptors = [
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                mol.GetNumHeavyAtoms(),
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcTPSA(mol),
                rdMolDescriptors.CalcNumRings(mol),
            ]
            
            # Handle NaN/inf values
            descriptors = [0.0 if (np.isnan(x) or np.isinf(x)) else float(x) for x in descriptors]
            return descriptors
            
        except Exception:
            return [0.0] * 10
    
    def featurize_molecules(self, smiles_list):
        """Extract molecular features."""
        logger.info(f"Featurizing {len(smiles_list)} molecules...")
        
        all_features = []
        for smiles in tqdm(smiles_list, desc="Featurizing"):
            features = self.get_molecular_descriptors(smiles)
            all_features.append(features)
        
        return np.array(all_features)
    
    def fit_scaler(self, features):
        """Fit the scaler."""
        self.scaler.fit(features)
        self.fitted = True
    
    def transform_features(self, features):
        """Transform features."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted first")
        return self.scaler.transform(features)

class PolymerDataset(torch_geometric.data.Dataset):
    """Simple dataset for polymer graphs."""
    
    def __init__(self, df, featurizer, target_cols=None, is_test=False):
        super().__init__()
        self.df = df
        self.featurizer = featurizer
        self.is_test = is_test
        self.target_cols = target_cols or TARGET_COLS
        
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

class SimpleGNN(nn.Module):
    """Simple GNN for polymer prediction."""
    
    def __init__(self, num_atom_features, hidden_channels=256, num_layers=4, num_targets=5):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_atom_features, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_channels, num_targets)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        
        x = global_mean_pool(x, batch)
        return self.out(x)

def masked_mse_loss(predictions, targets, masks):
    """MSE loss with masks."""
    masked_predictions = predictions * masks
    masked_targets = targets * masks
    
    loss = F.mse_loss(masked_predictions, masked_targets, reduction='none')
    loss = loss * masks
    
    valid_count = masks.sum()
    if valid_count > 0:
        return loss.sum() / valid_count
    else:
        return torch.tensor(0.0, device=predictions.device)

def train_epoch(model, loader, optimizer, device):
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
        loss = masked_mse_loss(predictions, batch.y, batch.mask)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs
    
    return total_loss / max(total_samples, 1)

@torch.no_grad()
def evaluate_model(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        if batch is None:
            continue
        
        batch = batch.to(device)
        predictions = model(batch)
        
        # Check if batch has targets and masks (validation data should have them)
        if hasattr(batch, 'y') and hasattr(batch, 'mask'):
            loss = masked_mse_loss(predictions, batch.y, batch.mask)
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
        else:
            # Skip batches without targets (shouldn't happen in validation)
            logger.warning("Batch missing targets or masks in evaluation")
            continue
    
    return total_loss / max(total_samples, 1)

@torch.no_grad()
def predict_with_model(model, loader, device):
    """Generate predictions."""
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

def load_data():
    """Load data."""
    logger.info("Loading data...")
    
    # Try different paths
    train_paths = [
        os.path.join(DATA_PATH, TRAIN_FILE),
        "info/train.csv",
        "train.csv"
    ]
    
    test_paths = [
        os.path.join(DATA_PATH, TEST_FILE),
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
        raise FileNotFoundError("Could not find data files")
    
    # Validate SMILES
    def is_valid_smiles(smiles):
        try:
            if not isinstance(smiles, str) or len(smiles.strip()) == 0:
                return False
            mol = Chem.MolFromSmiles(smiles.strip())
            return mol is not None and mol.GetNumAtoms() > 0
        except:
            return False
    
    train_valid = train_df['SMILES'].apply(is_valid_smiles)
    test_valid = test_df['SMILES'].apply(is_valid_smiles)
    
    logger.info(f"Valid SMILES - Train: {train_valid.sum()}/{len(train_df)}, Test: {test_valid.sum()}/{len(test_df)}")
    
    train_df = train_df[train_valid].reset_index(drop=True)
    test_df = test_df[test_valid].reset_index(drop=True)
    
    return train_df, test_df

def create_data_loaders(train_df, test_df, featurizer):
    """Create data loaders."""
    train_dataset = PolymerDataset(train_df, featurizer, is_test=False)
    test_dataset = PolymerDataset(test_df, featurizer, is_test=True)
    
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return Batch.from_data_list(batch)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    return train_loader, test_loader

def main():
    """Main pipeline."""
    logger.info("Starting Minimal Polymer Prediction Pipeline")
    logger.info(f"Device: {DEVICE}")
    
    try:
        # Load data
        train_df, test_df = load_data()
        
        # Initialize featurizer
        featurizer = SimpleFeaturizer()
        
        # Create data loaders
        train_loader, test_loader = create_data_loaders(train_df, test_df, featurizer)
        
        # Split for validation
        train_idx, val_idx = train_test_split(range(len(train_df)), test_size=0.2, random_state=RANDOM_SEED)
        
        train_subset = train_df.iloc[train_idx].reset_index(drop=True)
        val_subset = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Create validation dataset with targets (not test dataset)
        train_dataset_subset = PolymerDataset(train_subset, featurizer, is_test=False)
        val_dataset = PolymerDataset(val_subset, featurizer, is_test=False)  # Changed from is_test=True
        
        def collate_fn(batch):
            batch = [item for item in batch if item is not None]
            if len(batch) == 0:
                return None
            return Batch.from_data_list(batch)
        
        train_loader_subset = DataLoader(train_dataset_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        # Get sample for feature dimensions
        sample_batch = next(iter(train_loader_subset))
        if sample_batch is None:
            raise ValueError("No valid batches")
        
        num_atom_features = sample_batch.x.size(1)
        logger.info(f"Atom features: {num_atom_features}")
        
        # Initialize model
        model = SimpleGNN(
            num_atom_features=num_atom_features,
            hidden_channels=HIDDEN_CHANNELS,
            num_layers=NUM_GCN_LAYERS,
            num_targets=len(TARGET_COLS)
        ).to(DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader_subset, optimizer, DEVICE)
            val_loss = evaluate_model(model, val_loader, DEVICE)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Load best model and predict
        model.load_state_dict(torch.load('best_model.pt'))
        test_ids, predictions = predict_with_model(model, test_loader, DEVICE)
        
        # Create submission
        submission = pd.DataFrame({'id': test_ids})
        for i, col in enumerate(TARGET_COLS):
            if i < predictions.shape[1]:
                submission[col] = predictions[:, i]
            else:
                submission[col] = 0.0
        
        # Validate and save
        for col in TARGET_COLS:
            if submission[col].isna().any():
                submission[col].fillna(0, inplace=True)
            if np.isinf(submission[col]).any():
                submission[col] = np.clip(submission[col], -1e6, 1e6)
        
        submission.to_csv(SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {SUBMISSION_FILE}")
        logger.info(f"Submission shape: {submission.shape}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create dummy submission
        try:
            if 'test_df' in locals():
                dummy_submission = pd.DataFrame({'id': test_df['id']})
                for col in TARGET_COLS:
                    dummy_submission[col] = 0.0
                dummy_submission.to_csv('dummy_' + SUBMISSION_FILE, index=False)
                logger.info(f"Dummy submission saved")
        except:
            logger.error("Failed to create dummy submission")
        
        raise

if __name__ == "__main__":
    main()