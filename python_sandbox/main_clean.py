import sys
import subprocess

def install_packages():
    packages = [
        'torch',
        'torch-geometric',
        'rdkit',
        'xgboost',
        'lightgbm',
        'catboost',
        'optuna',
        'scikit-learn',
        'pandas',
        'numpy',
        'tqdm'
    ]
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', package])
        except Exception as e:
            print(f"Warning: Failed to install {package}: {e}")

install_packages()

# Step 2: Import libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader  # Fix: Use PyTorch Geometric DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.data  # Fix: Add missing torch_geometric.data import
from tqdm import tqdm
import gc
import logging

# Import existing production components
sys.path.append('src')
from polymer_prediction.data.dataset import PolymerDataset as ProductionPolymerDataset
from polymer_prediction.training.trainer import (
    masked_mse_loss, 
    train_one_epoch, 
    evaluate, 
    predict
)
from polymer_prediction.preprocessing.featurization import smiles_to_graph
from polymer_prediction.config.config import CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Enhanced to use existing production config
class Config:
    def __init__(self):
        # Use existing config as base
        self.DEVICE = CONFIG.DEVICE
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.LEARNING_RATE = CONFIG.LEARNING_RATE
        self.HIDDEN_CHANNELS = CONFIG.HIDDEN_CHANNELS
        self.NUM_GCN_LAYERS = CONFIG.NUM_GCN_LAYERS
        self.NUM_EPOCHS = CONFIG.NUM_EPOCHS
        
        # Additional configuration
        self.DATA_PATH = 'info'  # Updated for local data
        self.TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.N_FOLDS = 5
        self.RANDOM_STATE = 42
        self.MORGAN_BITS = 2048
        self.OPTUNA_TRIALS = 15
        self.GCN_EPOCHS = 50
        
        # CPU optimizations
        if self.DEVICE.type == 'cpu':
            self.BATCH_SIZE = 16
            self.GCN_EPOCHS = 50

config = Config()
np.random.seed(config.RANDOM_STATE)
torch.manual_seed(config.RANDOM_STATE)

# Step 3: Data Loading with SMILES validation
def validate_smiles(smiles):
    """Validate a SMILES string using RDKit."""
    if pd.isna(smiles) or smiles == '':
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def load_data():
    train_df = pd.read_csv(f'{config.DATA_PATH}/train.csv')
    test_df = pd.read_csv(f'{config.DATA_PATH}/test.csv')
    
    logger.info(f"Data loaded: Train {train_df.shape}, Test {test_df.shape}")
    
    # Validate SMILES strings and filter out invalid ones
    logger.info("Validating SMILES strings...")
    
    # Check for invalid SMILES in training data
    train_valid_mask = train_df['SMILES'].apply(validate_smiles)
    invalid_train_count = (~train_valid_mask).sum()
    if invalid_train_count > 0:
        logger.warning(f"Found {invalid_train_count} invalid SMILES in training data, filtering them out")
        train_df = train_df[train_valid_mask].reset_index(drop=True)
    
    # Check for invalid SMILES in test data
    test_valid_mask = test_df['SMILES'].apply(validate_smiles)
    invalid_test_count = (~test_valid_mask).sum()
    if invalid_test_count > 0:
        logger.warning(f"Found {invalid_test_count} invalid SMILES in test data")
        # For test data, we need to keep all rows but mark invalid ones
        test_df['valid_smiles'] = test_valid_mask
    else:
        test_df['valid_smiles'] = True
    
    logger.info(f"After validation: Train {train_df.shape}, Test {test_df.shape}")
    return train_df, test_df

# Step 4: Enhanced Feature Extraction (from CPU notebook, extended)
class FeatureExtractor:
    def __init__(self):
        self.morgan_gen = GetMorganGenerator(radius=2, fpSize=config.MORGAN_BITS)

    def extract(self, smiles_list):
        features = []
        for smiles in tqdm(smiles_list):
            # Skip invalid SMILES
            if not validate_smiles(smiles):
                features.append(np.zeros(60 + config.MORGAN_BITS))
                continue
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                features.append(np.zeros(60 + config.MORGAN_BITS))
                continue

            # Extended RDKit descriptors
            desc = [
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                Descriptors.NumAromaticRings(mol), Descriptors.NumSaturatedRings(mol), Descriptors.NumAliphaticRings(mol),
                Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol), Descriptors.NumHeteroatoms(mol),
                Descriptors.RingCount(mol), Descriptors.BertzCT(mol), Descriptors.BalabanJ(mol),
                Descriptors.Chi0v(mol), Descriptors.Chi1v(mol), Descriptors.Chi2v(mol), Descriptors.Chi3v(mol),
                Descriptors.Chi4v(mol), Descriptors.Kappa1(mol), Descriptors.Kappa2(mol), Descriptors.Kappa3(mol),
                mol.GetNumAtoms(), mol.GetNumBonds(), rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcNumHBD(mol), rdMolDescriptors.CalcNumHBA(mol),
                rdMolDescriptors.CalcNumRings(mol), rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumSaturatedRings(mol), rdMolDescriptors.CalcNumAliphaticRings(mol),
                smiles.count('*'), len(smiles), smiles.count('C'), smiles.count('N'), smiles.count('O'),
                smiles.count('='), smiles.count('#'), smiles.count('('),
                # Additional: Labute ASA, Exact Mass, etc.
                Descriptors.LabuteASA(mol), Descriptors.ExactMolWt(mol), Descriptors.MaxPartialCharge(mol),
                Descriptors.MinPartialCharge(mol), Descriptors.MaxAbsPartialCharge(mol),
                Descriptors.MinAbsPartialCharge(mol), Descriptors.MolMR(mol), Descriptors.VSA_EState1(mol),
                Descriptors.VSA_EState2(mol), Descriptors.VSA_EState3(mol), Descriptors.VSA_EState4(mol),
                Descriptors.VSA_EState5(mol), Descriptors.VSA_EState6(mol), Descriptors.VSA_EState7(mol),
                Descriptors.VSA_EState8(mol), Descriptors.VSA_EState9(mol), Descriptors.VSA_EState10(mol)
            ]
            desc = desc[:60]  # Limit to 60 for efficiency

            fp = self.morgan_gen.GetFingerprint(mol)
            fp_arr = np.array([fp.GetBit(i) for i in range(config.MORGAN_BITS)])

            feat = np.concatenate([desc, fp_arr])
            features.append(np.nan_to_num(feat, nan=0.0))

        return np.array(features, dtype=np.float32)

# Step 5: GCN Model (from fork notebook, enhanced with dropout and more layers)
class PolymerGCN(nn.Module):
    def __init__(self, num_atom_features, hidden_channels=128, num_gcn_layers=8):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(num_atom_features, hidden_channels)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_channels, len(config.TARGET_COLS))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.out(x)

def train_gcn(train_df, test_df):
    # Filter test data to only valid SMILES for GCN training
    test_df_valid = test_df[test_df['valid_smiles']].copy() if 'valid_smiles' in test_df.columns else test_df.copy()
    
    # Create datasets using production PolymerDataset
    dataset = ProductionPolymerDataset(train_df, target_cols=config.TARGET_COLS, is_test=False)
    test_dataset = ProductionPolymerDataset(test_df_valid, target_cols=config.TARGET_COLS, is_test=True)
    
    # Create safe DataLoaders that handle None values
    def create_safe_dataloader(dataset, batch_size, shuffle=False):
        """Create a DataLoader with proper error handling for invalid SMILES."""
        from torch.utils.data import DataLoader as TorchDataLoader
        
        def collate_fn(batch):
            # Filter out None values (failed SMILES parsing)
            valid_batch = [item for item in batch if item is not None]
            if len(valid_batch) == 0:
                return None
            return torch_geometric.data.Batch.from_data_list(valid_batch)
        
        return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              collate_fn=collate_fn, drop_last=False)
    
    loader = create_safe_dataloader(dataset, config.BATCH_SIZE, shuffle=True)
    test_loader = create_safe_dataloader(test_dataset, config.BATCH_SIZE, shuffle=False)

    # Get number of atom features from a sample
    sample_data = None
    for data in dataset:
        if data is not None:
            sample_data = data
            break
    
    if sample_data is None:
        raise ValueError("No valid molecular graphs found in dataset")
    
    num_atom_features = sample_data.x.size(1)
    logger.info(f"Number of atom features: {num_atom_features}")

    model = PolymerGCN(num_atom_features=num_atom_features).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    logger.info("Starting GCN training...")
    for epoch in range(config.GCN_EPOCHS):
        # Use existing training function from production components
        avg_loss = train_one_epoch(model, loader, optimizer, config.DEVICE)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    # Generate test predictions using existing predict function
    logger.info("Generating GCN test predictions...")
    test_ids, test_preds = predict(model, test_loader, config.DEVICE)
    
    # Handle invalid SMILES in test data by creating full prediction array
    if 'valid_smiles' in test_df.columns:
        full_test_preds = np.zeros((len(test_df), len(config.TARGET_COLS)))
        valid_indices = test_df[test_df['valid_smiles']].index.tolist()
        
        # Fill in predictions for valid SMILES
        for i, pred in enumerate(test_preds):
            if i < len(valid_indices):
                full_test_preds[valid_indices[i]] = pred
        
        return full_test_preds
    else:
        return test_preds

def train_tree_ensemble(train_df, test_df, extractor):
    X_train = extractor.extract(train_df['SMILES'])
    X_test = extractor.extract(test_df['SMILES'])

    ensemble_preds = np.zeros((len(test_df), len(config.TARGET_COLS)))
    
    for i, target in enumerate(config.TARGET_COLS):
        logger.info(f"Training tree ensemble for target: {target}")
        
        # Get target values, fill missing with median
        y = train_df[target].fillna(train_df[target].median()).values
        
        # Simple ensemble with LightGBM, XGBoost, and CatBoost
        models = []
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=config.RANDOM_STATE,
            verbose=-1
        )
        lgb_model.fit(X_train, y)
        models.append(lgb_model)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=config.RANDOM_STATE,
            verbosity=0
        )
        xgb_model.fit(X_train, y)
        models.append(xgb_model)
        
        # CatBoost
        cat_model = cb.CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_state=config.RANDOM_STATE,
            verbose=False
        )
        cat_model.fit(X_train, y)
        models.append(cat_model)
        
        # Average predictions from all models
        target_preds = np.zeros(len(test_df))
        for model in models:
            target_preds += model.predict(X_test)
        target_preds /= len(models)
        
        ensemble_preds[:, i] = target_preds

    return ensemble_preds

# Step 7: Main Pipeline with Simple Ensemble
def main():
    logger.info("Starting main pipeline...")
    train_df, test_df = load_data()
    extractor = FeatureExtractor()

    # Get predictions from both models
    logger.info("Training GCN model...")
    gcn_preds = train_gcn(train_df, test_df)
    
    logger.info("Training tree ensemble...")
    tree_preds = train_tree_ensemble(train_df, test_df, extractor)

    # Simple ensemble: average the predictions
    logger.info("Combining predictions...")
    final_preds = (gcn_preds + tree_preds) / 2.0

    # Create submission
    submission = pd.DataFrame({'id': test_df['id']})
    for i, col in enumerate(config.TARGET_COLS):
        submission[col] = final_preds[:, i]
    
    submission.to_csv('submission.csv', index=False)
    logger.info("Submission saved to submission.csv!")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info("First few predictions:")
    logger.info(submission.head())

if __name__ == "__main__":
    main()