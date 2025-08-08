"""
Improved Polymer Prediction Pipeline with Production Components Integration

This module integrates existing production-ready components from the repository
to create a robust polymer prediction system.
"""

import sys
import subprocess
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader  # Fix: Use PyTorch Geometric DataLoader
import torch_geometric.data
from tqdm import tqdm
import gc
import logging
from pathlib import Path

# Import existing production components
sys.path.append('src')
from polymer_prediction.data.dataset import PolymerDataset
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

def install_packages():
    """Install required packages if not available."""
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
            logger.warning(f"Failed to install {package}: {e}")

# Install packages
install_packages()

class ImprovedConfig:
    """Enhanced configuration class that extends the existing CONFIG."""
    
    def __init__(self):
        # Use existing config as base
        self.DEVICE = CONFIG.DEVICE
        self.BATCH_SIZE = CONFIG.BATCH_SIZE
        self.LEARNING_RATE = CONFIG.LEARNING_RATE
        self.HIDDEN_CHANNELS = CONFIG.HIDDEN_CHANNELS
        self.NUM_GCN_LAYERS = CONFIG.NUM_GCN_LAYERS
        self.NUM_EPOCHS = CONFIG.NUM_EPOCHS
        
        # Additional configuration for the improved pipeline
        self.DATA_PATH = 'info'  # Updated for local data
        self.TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.N_FOLDS = 5
        self.RANDOM_STATE = 42
        self.MORGAN_BITS = 2048
        self.OPTUNA_TRIALS = 15
        
        # CPU optimizations
        if self.DEVICE.type == 'cpu':
            self.BATCH_SIZE = 16
            self.NUM_EPOCHS = 50

config = ImprovedConfig()
np.random.seed(config.RANDOM_STATE)
torch.manual_seed(config.RANDOM_STATE)

def load_data():
    """Load training and test data."""
    try:
        train_df = pd.read_csv(f'{config.DATA_PATH}/train.csv')
        test_df = pd.read_csv(f'{config.DATA_PATH}/test.csv')
        logger.info(f"Data loaded: Train {train_df.shape}, Test {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.info("Please ensure train.csv and test.csv are in the 'info' directory")
        raise

class EnhancedFeatureExtractor:
    """Enhanced feature extractor using RDKit descriptors and Morgan fingerprints."""
    
    def __init__(self):
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        
        self.morgan_gen = GetMorganGenerator(radius=2, fpSize=config.MORGAN_BITS)
        self.Chem = Chem
        self.Descriptors = Descriptors
        self.rdMolDescriptors = rdMolDescriptors

    def extract(self, smiles_list):
        """Extract features from SMILES strings."""
        features = []
        for smiles in tqdm(smiles_list, desc="Extracting features"):
            mol = self.Chem.MolFromSmiles(smiles)
            if mol is None:
                features.append(np.zeros(60 + config.MORGAN_BITS))
                continue

            # Extended RDKit descriptors
            desc = [
                self.Descriptors.MolWt(mol), self.Descriptors.MolLogP(mol), 
                self.Descriptors.TPSA(mol), self.Descriptors.NumRotatableBonds(mol), 
                self.Descriptors.NumHDonors(mol), self.Descriptors.NumHAcceptors(mol),
                self.Descriptors.NumAromaticRings(mol), self.Descriptors.NumSaturatedRings(mol), 
                self.Descriptors.NumAliphaticRings(mol), self.Descriptors.FractionCSP3(mol), 
                self.Descriptors.HeavyAtomCount(mol), self.Descriptors.NumHeteroatoms(mol),
                self.Descriptors.RingCount(mol), self.Descriptors.BertzCT(mol), 
                self.Descriptors.BalabanJ(mol), self.Descriptors.Chi0v(mol), 
                self.Descriptors.Chi1v(mol), self.Descriptors.Chi2v(mol), 
                self.Descriptors.Chi3v(mol), self.Descriptors.Chi4v(mol), 
                self.Descriptors.Kappa1(mol), self.Descriptors.Kappa2(mol), 
                self.Descriptors.Kappa3(mol), mol.GetNumAtoms(), mol.GetNumBonds(), 
                self.rdMolDescriptors.CalcNumRotatableBonds(mol),
                self.rdMolDescriptors.CalcNumHBD(mol), self.rdMolDescriptors.CalcNumHBA(mol),
                self.rdMolDescriptors.CalcNumRings(mol), self.rdMolDescriptors.CalcNumAromaticRings(mol),
                self.rdMolDescriptors.CalcNumSaturatedRings(mol), self.rdMolDescriptors.CalcNumAliphaticRings(mol),
                smiles.count('*'), len(smiles), smiles.count('C'), smiles.count('N'), 
                smiles.count('O'), smiles.count('='), smiles.count('#'), smiles.count('('),
                # Additional descriptors
                self.Descriptors.LabuteASA(mol), self.Descriptors.ExactMolWt(mol), 
                self.Descriptors.MaxPartialCharge(mol), self.Descriptors.MinPartialCharge(mol), 
                self.Descriptors.MaxAbsPartialCharge(mol), self.Descriptors.MinAbsPartialCharge(mol), 
                self.Descriptors.MolMR(mol), self.Descriptors.VSA_EState1(mol),
                self.Descriptors.VSA_EState2(mol), self.Descriptors.VSA_EState3(mol), 
                self.Descriptors.VSA_EState4(mol), self.Descriptors.VSA_EState5(mol), 
                self.Descriptors.VSA_EState6(mol), self.Descriptors.VSA_EState7(mol),
                self.Descriptors.VSA_EState8(mol), self.Descriptors.VSA_EState9(mol), 
                self.Descriptors.VSA_EState10(mol)
            ]
            desc = desc[:60]  # Limit to 60 for efficiency

            # Morgan fingerprint
            fp = self.morgan_gen.GetFingerprint(mol)
            fp_arr = np.array([fp.GetBit(i) for i in range(config.MORGAN_BITS)])

            feat = np.concatenate([desc, fp_arr])
            features.append(np.nan_to_num(feat, nan=0.0))

        return np.array(features, dtype=np.float32)

class PolymerGCN(nn.Module):
    """Enhanced GCN model using existing architecture patterns."""
    
    def __init__(self, num_atom_features, hidden_channels=None, num_gcn_layers=None):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        # Use config values
        hidden_channels = hidden_channels or config.HIDDEN_CHANNELS
        num_gcn_layers = num_gcn_layers or config.NUM_GCN_LAYERS
        
        self.convs = nn.ModuleList([GCNConv(num_atom_features, hidden_channels)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_channels, len(config.TARGET_COLS))
        self.global_mean_pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = self.dropout(x)
        
        x = self.global_mean_pool(x, batch)
        return self.out(x)

def create_safe_dataloader(dataset, batch_size, shuffle=False):
    """Create a DataLoader with proper error handling for invalid SMILES."""
    def collate_fn(batch):
        # Filter out None values (failed SMILES parsing)
        valid_batch = [item for item in batch if item is not None]
        if len(valid_batch) == 0:
            return None
        return torch_geometric.data.Batch.from_data_list(valid_batch)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     collate_fn=collate_fn, drop_last=False)

def train_gcn_with_existing_components(train_df, test_df):
    """Train GCN using existing production components."""
    logger.info("Training GCN with existing production components...")
    
    # Create datasets using existing PolymerDataset
    dataset = PolymerDataset(train_df, target_cols=config.TARGET_COLS, is_test=False)
    test_dataset = PolymerDataset(test_df, target_cols=config.TARGET_COLS, is_test=True)
    
    # Create DataLoaders with proper error handling
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

    # Initialize model
    model = PolymerGCN(num_atom_features=num_atom_features).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    logger.info("Starting GCN training...")
    for epoch in range(config.NUM_EPOCHS):
        # Use existing training function
        avg_loss = train_one_epoch(model, loader, optimizer, config.DEVICE)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    # Generate test predictions using existing predict function
    logger.info("Generating GCN test predictions...")
    test_ids, test_preds = predict(model, test_loader, config.DEVICE)
    
    return test_preds

def main():
    """Main pipeline using integrated production components."""
    logger.info("Starting improved polymer prediction pipeline...")
    
    try:
        # Load data
        train_df, test_df = load_data()
        
        # Train GCN with existing components
        gcn_preds = train_gcn_with_existing_components(train_df, test_df)
        
        # Create submission (simplified for now)
        submission = pd.DataFrame({'id': test_df['id']})
        for i, col in enumerate(config.TARGET_COLS):
            submission[col] = gcn_preds[:, i]
        
        submission.to_csv('submission_improved.csv', index=False)
        logger.info("Improved submission generated successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def test_integration():
    """Test the integration of existing components."""
    logger.info("Testing integration of existing production components...")
    
    # Create sample data
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'SMILES': [
            'CCO',  # Ethanol
            'c1ccccc1',  # Benzene
            'CC(C)O',  # Isopropanol
            'C1=CC=CC=C1O',  # Phenol
            'CCCCCCCC'  # Octane
        ],
        'Tg': [1.0, 2.0, None, 3.0, 4.0],
        'FFV': [0.1, 0.2, 0.3, None, 0.5],
        'Tc': [10.0, None, 30.0, 40.0, 50.0],
        'Density': [0.8, 0.9, 1.0, 1.1, None],
        'Rg': [2.0, 3.0, 4.0, 5.0, 6.0]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    try:
        # Test dataset creation with existing PolymerDataset
        dataset = PolymerDataset(sample_df, target_cols=config.TARGET_COLS, is_test=False)
        logger.info(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test DataLoader
        loader = create_safe_dataloader(dataset, batch_size=2, shuffle=False)
        
        # Test model with existing components
        sample_data = None
        for data in dataset:
            if data is not None:
                sample_data = data
                break
        
        if sample_data is not None:
            num_atom_features = sample_data.x.size(1)
            model = PolymerGCN(num_atom_features=num_atom_features)
            model.eval()
            
            for batch in loader:
                if batch is not None:
                    with torch.no_grad():
                        output = model(batch)
                    logger.info(f"Model output shape: {output.shape}")
                    break
            
            logger.info("Integration test completed successfully!")
            return True
        else:
            logger.error("No valid samples found in dataset")
            return False
            
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test integration first
    if test_integration():
        logger.info("Integration test passed! Running main pipeline...")
        main()
    else:
        logger.error("Integration test failed! Please fix issues before running main pipeline.")