# NeurIPS Open Polymer Prediction 2025 - GPU-Optimized Advanced Solution
# Memory-efficient implementation for RTX 2060 (6GB VRAM)

import os
import sys
import random
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import math

# Data handling
import pandas as pd
import numpy as np
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# PyTorch and PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.pool import Set2Set
from torch_geometric.utils import degree

# RDKit for chemistry
from rdkit import Chem
from rdkit.Chem import rdchem, Descriptors, Crippen, Lipinski
from rdkit import RDLogger

# Suppress warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class Config:
    """Enhanced configuration with GPU optimization and best practices."""
    
    def __init__(self):
        # Device configuration
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.USE_MIXED_PRECISION = torch.cuda.is_available()
        
        # Model architecture - Based on literature review
        self.MODEL_TYPE = 'GAT'  # GAT often outperforms GCN on molecular data
        self.NUM_EPOCHS = 200  # Increased for better convergence
        self.BATCH_SIZE = 64 if torch.cuda.is_available() else 32  # GPU optimized
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-4
        
        # GAT-specific hyperparameters (from literature)
        self.HIDDEN_CHANNELS = 256  # Increased capacity
        self.NUM_ATTENTION_HEADS = 8
        self.NUM_LAYERS = 4
        self.DROPOUT = 0.2
        
        # Advanced training features
        self.USE_GRADIENT_CHECKPOINTING = True
        self.EARLY_STOPPING_PATIENCE = 20
        self.VAL_SPLIT_FRACTION = 0.15  # More data for training
        self.SEED = 42
        
        # Memory optimization
        self.PIN_MEMORY = torch.cuda.is_available()
        self.NUM_WORKERS = 4 if torch.cuda.is_available() else 0
        
        # Target properties
        self.TARGET_PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Learning rate scheduling
        self.USE_SCHEDULER = True
        self.SCHEDULER_PATIENCE = 10
        self.SCHEDULER_FACTOR = 0.7

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (might slightly reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_molecular_features(mol):
    """Enhanced molecular feature extraction with more descriptors."""
    if mol is None:
        return np.zeros(13)  # Return zero array for invalid molecules
    
    features = []
    
    # Basic descriptors
    features.append(mol.GetNumAtoms())
    features.append(mol.GetNumBonds())
    features.append(Descriptors.MolWt(mol))
    features.append(Descriptors.MolLogP(mol))
    features.append(Descriptors.NumHDonors(mol))
    features.append(Descriptors.NumHAcceptors(mol))
    features.append(Descriptors.TPSA(mol))
    features.append(Descriptors.NumRotatableBonds(mol))
    features.append(Descriptors.NumAromaticRings(mol))
    features.append(Descriptors.NumSaturatedRings(mol))
    features.append(Descriptors.FractionCsp3(mol))
    features.append(Descriptors.BalabanJ(mol))
    features.append(Descriptors.BertzCT(mol))
    
    return np.array(features, dtype=np.float32)

def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert SMILES to PyTorch Geometric Data with enhanced features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Atom features (more comprehensive)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetNumRadicalElectrons(),
            atom.GetTotalNumHs(),
            int(atom.IsInRing()),
            int(atom.IsInRingSize(3)),
            int(atom.IsInRingSize(4)),
            int(atom.IsInRingSize(5)),
            int(atom.IsInRingSize(6)),
            int(atom.IsInRingSize(7)),
            int(atom.IsInRingSize(8)),
        ]
        atom_features.append(features)
    
    # Bond features and edge indices
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        bond_feature = [
            int(bond.GetBondType()),
            int(bond.GetIsAromatic()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
        ]
        
        # Add both directions for undirected graph
        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([bond_feature, bond_feature])
    
    # Convert to tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Global molecular features
    global_features = get_molecular_features(mol)
    global_features = torch.tensor(global_features, dtype=torch.float).unsqueeze(0)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=global_features)

class PolymerDataset(Dataset):
    """Enhanced dataset with better preprocessing and memory management."""
    
    def __init__(self, df: pd.DataFrame, target_cols: List[str], transform=None):
        super().__init__(transform)
        self.df = df.copy()
        self.target_cols = target_cols
        
        # Preprocess data
        self._preprocess_data()
        
        # Convert SMILES to graphs (with progress bar)
        print("Converting SMILES to molecular graphs...")
        self.graphs = []
        self.valid_indices = []
        
        for idx, smiles in tqdm(enumerate(self.df['SMILES']), total=len(self.df)):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                self.valid_indices.append(idx)
        
        # Update dataframe to only include valid molecules
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        print(f"Successfully converted {len(self.graphs)} out of {len(df)} molecules")
    
    def _preprocess_data(self):
        """Preprocess target values with better missing value handling."""
        # Handle missing values with median imputation (better than mean for skewed data)
        for col in self.target_cols:
            if col in self.df.columns:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                
                # Standardize targets for better training stability
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                self.df[col] = (self.df[col] - mean_val) / (std_val + 1e-8)
    
    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        graph = self.graphs[idx].clone()
        
        # Add targets
        targets = []
        for col in self.target_cols:
            if col in self.df.columns:
                targets.append(self.df.iloc[idx][col])
            else:
                targets.append(0.0)  # Default value for missing targets
        
        graph.y = torch.tensor(targets, dtype=torch.float)
        return graph

class AdvancedGATModel(nn.Module):
    """Enhanced GAT model with memory optimization and best practices."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Input dimensions
        self.node_dim = 14  # Enhanced atom features
        self.edge_dim = 4   # Bond features
        self.global_dim = 13  # Global molecular features
        
        # GAT layers with multi-head attention
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            GATConv(
                self.node_dim, 
                config.HIDDEN_CHANNELS // config.NUM_ATTENTION_HEADS,
                heads=config.NUM_ATTENTION_HEADS,
                dropout=config.DROPOUT,
                edge_dim=self.edge_dim,
                concat=True
            )
        )
        
        # Hidden layers
        for _ in range(config.NUM_LAYERS - 2):
            self.gat_layers.append(
                GATConv(
                    config.HIDDEN_CHANNELS,
                    config.HIDDEN_CHANNELS // config.NUM_ATTENTION_HEADS,
                    heads=config.NUM_ATTENTION_HEADS,
                    dropout=config.DROPOUT,
                    edge_dim=self.edge_dim,
                    concat=True
                )
            )
        
        # Final layer (single head)
        self.gat_layers.append(
            GATConv(
                config.HIDDEN_CHANNELS,
                config.HIDDEN_CHANNELS,
                heads=1,
                dropout=config.DROPOUT,
                edge_dim=self.edge_dim,
                concat=False
            )
        )
        
        # Pooling layers for different aggregations
        self.set2set = Set2Set(config.HIDDEN_CHANNELS, processing_steps=3)
        
        # Prediction head with global features
        combined_dim = config.HIDDEN_CHANNELS * 2 + self.global_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, config.HIDDEN_CHANNELS),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_CHANNELS // 2, len(config.TARGET_PROPERTIES))
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        # GAT forward pass with residual connections
        h = x
        for i, gat_layer in enumerate(self.gat_layers):
            if self.config.USE_GRADIENT_CHECKPOINTING and self.training:
                h_new = torch.utils.checkpoint.checkpoint(gat_layer, h, edge_index, edge_attr)
            else:
                h_new = gat_layer(h, edge_index, edge_attr)
            
            h_new = F.elu(h_new)
            
            # Residual connection (if dimensions match)
            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
        
        # Graph-level pooling using Set2Set (more expressive than simple pooling)
        graph_embedding = self.set2set(h, batch_idx)
        
        # Combine with global molecular features
        if hasattr(batch, 'global_features'):
            global_features = batch.global_features.view(graph_embedding.size(0), -1)
            combined_features = torch.cat([graph_embedding, global_features], dim=1)
        else:
            combined_features = graph_embedding
        
        # Prediction
        out = self.predictor(combined_features)
        return out

def create_data_loaders(dataset, config: Config):
    """Create optimized data loaders with memory efficiency."""
    # Split dataset
    val_size = int(len(dataset) * config.VAL_SPLIT_FRACTION)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    # Create data loaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.NUM_WORKERS > 0,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.NUM_WORKERS > 0
    )
    
    return train_loader, val_loader

def weighted_mae_loss(predictions, targets, weights=None):
    """Compute weighted MAE loss as used in the competition."""
    if weights is None:
        weights = torch.ones_like(targets)
    
    # Handle missing values (NaN)
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    mae = torch.abs(predictions - targets)
    weighted_mae = (mae * weights * mask.float()).sum() / (weights * mask.float()).sum()
    return weighted_mae

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict({k: v.to(model.device) for k, v in self.best_weights.items()})
            return True
        return False

def train_epoch(model, train_loader, optimizer, scaler, config):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(config.DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        if config.USE_MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                predictions = model(batch)
                loss = weighted_mae_loss(predictions, batch.y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = model(batch)
            loss = weighted_mae_loss(predictions, batch.y)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, config):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(config.DEVICE, non_blocking=True)
            
            if config.USE_MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    predictions = model(batch)
                    loss = weighted_mae_loss(predictions, batch.y)
            else:
                predictions = model(batch)
                loss = weighted_mae_loss(predictions, batch.y)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def main():
    """Main training function."""
    # Configuration
    config = Config()
    set_seed(config.SEED)
    
    print(f"Using device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Mixed Precision: {config.USE_MIXED_PRECISION}")
    print(f"Gradient Checkpointing: {config.USE_GRADIENT_CHECKPOINTING}")
    
    # Load data
    print("Loading data...")
    try:
        train_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
        test_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
    except FileNotFoundError:
        print("Competition data not found. Please ensure the data files are available.")
        return
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PolymerDataset(train_df, config.TARGET_PROPERTIES)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_dataset, config)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    model = AdvancedGATModel(config).to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    if config.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=config.SCHEDULER_FACTOR, 
            patience=config.SCHEDULER_PATIENCE,
            verbose=True
        )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, config)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, config)
        
        # Update scheduler
        if config.USE_SCHEDULER:
            scheduler.step(val_loss)
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        if epoch % 10 == 0 or epoch == config.NUM_EPOCHS - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("Training completed!")
    
    # Generate predictions for test set if available
    if len(test_df) > 0:
        print("Generating test predictions...")
        test_dataset = PolymerDataset(test_df, config.TARGET_PROPERTIES)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(config.DEVICE, non_blocking=True)
                
                if config.USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        pred = model(batch)
                else:
                    pred = model(batch)
                
                predictions.append(pred.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Create submission
        submission_df = test_df[['id']].copy()
        for i, col in enumerate(config.TARGET_PROPERTIES):
            submission_df[col] = predictions[:, i]
        
        submission_df.to_csv('submission.csv', index=False)
        print("Submission saved to submission.csv")
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Final validation loss: {val_losses[-1]:.4f}")

if __name__ == "__main__":
    main()