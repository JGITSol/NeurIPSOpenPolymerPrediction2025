# SOTA Starter Code for Polymer Property Prediction
#
# This script implements a Graph Neural Network (GNN) for predicting molecular
# properties from SMILES strings. It's a self-contained, high-quality
# starting point for a Kaggle competition.
#
# Key Libraries:
# - PyTorch: Core deep learning framework.
# - PyTorch Geometric (PyG): The de-facto library for GNNs in PyTorch.
# - RDKit: The essential toolkit for cheminformatics.
# - Pandas/Numpy: For data manipulation.
#
# To Run (after installing dependencies):
# pip install torch torch-geometric rdkit-pypi pandas numpy scikit-learn

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import io

# RDKit for chemistry
from rdkit import Chem
from rdkit.Chem import AllChem

# PyTorch and PyG
import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Module, Sequential
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# --- 1. CONFIGURATION ---
# All hyperparameters and settings are centralized here for easy tuning.
class Config:
    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.DEVICE}")

        # Model hyperparameters
        self.NUM_EPOCHS = 50
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-5
        self.HIDDEN_CHANNELS = 128
        self.NUM_GCN_LAYERS = 3
        
        # Data
        self.TEST_SPLIT_FRACTION = 0.2
        self.SEED = 42

CONFIG = Config()

# --- 2. FEATURIZATION ---
# This is the most critical part: converting a SMILES string into a graph.

def get_atom_features(atom):
    """ Extracts features for a single atom. """
    # These features are standard choices for molecular GNNs.
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        atom.GetIsAromatic(),
        atom.GetChiralTag().real,
    ]
    # One-hot encode atomic number (example for C, O, N)
    # A more robust implementation would handle all possible elements.
    atomic_num_one_hot = [0] * 3
    if atom.GetAtomicNum() == 6: atomic_num_one_hot[0] = 1
    elif atom.GetAtomicNum() == 8: atomic_num_one_hot[1] = 1
    elif atom.GetAtomicNum() == 7: atomic_num_one_hot[2] = 1
    
    return features + atomic_num_one_hot

def smiles_to_graph(smiles_string):
    """Converts a SMILES string to a PyG Data object."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None
    mol = Chem.AddHs(mol) # Add explicit hydrogens

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
            edge_indices.append((j, i)) # Graph must be undirected

            # Example bond features
            bond_type = bond.GetBondTypeAsDouble()
            is_in_ring = bond.IsInRing()
            edge_attrs.append([bond_type, is_in_ring])
            edge_attrs.append([bond_type, is_in_ring])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Handle molecules with no bonds (e.g., single atoms)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Store number of features for model instantiation later
    data.num_atom_features = x.size(1)

    return data

# --- 3. PYTORCH GEOMETRIC DATASET ---
# A custom dataset class to handle on-the-fly conversion of SMILES to graphs.

class PolymerDataset(Dataset):
    def __init__(self, df, target_col):
        super().__init__()
        self.df = df
        self.smiles_list = df['smiles'].tolist()
        self.targets = df[target_col].tolist()
        self.cache = {} # Cache graphs to avoid re-computing

    def len(self):
        return len(self.df)

    def get(self, idx):
        if idx in self.cache:
            data = self.cache[idx]
        else:
            smiles = self.smiles_list[idx]
            data = smiles_to_graph(smiles)
            if data is None: # Handle RDKit parsing errors
                return None
            self.cache[idx] = data

        # Add target value to the data object
        data.y = torch.tensor([self.targets[idx]], dtype=torch.float)
        return data

# --- 4. GRAPH NEURAL NETWORK MODEL ---
# A GCN-based model for graph regression.

class PolymerGCN(Module):
    def __init__(self, num_atom_features, hidden_channels, num_gcn_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(num_atom_features, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))

        # Output layer
        self.out = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Message passing through GCN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        # Readout phase: Aggregate node features into a single graph-level representation
        x = global_mean_pool(x, batch)
        
        # Final prediction
        return self.out(x)

# --- 5. TRAINING & EVALUATION LOOPS ---

def train_one_epoch(model, loader, optimizer, loss_fn):
    """Performs one full training pass over the dataset."""
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training", leave=False):
        data = data.to(CONFIG.DEVICE)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    """Evaluates the model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    for data in tqdm(loader, desc="Evaluating", leave=False):
        data = data.to(CONFIG.DEVICE)
        out = model(data)
        loss = loss_fn(out, data.y)
        total_loss += loss.item() * data.num_graphs
        all_preds.append(out.cpu())
        all_targets.append(data.y.cpu())

    avg_loss = total_loss / len(loader.dataset)
    
    # Calculate RMSE
    preds = torch.cat(all_preds).numpy().flatten()
    targets = torch.cat(all_targets).numpy().flatten()
    rmse = np.sqrt(np.mean((preds - targets)**2))
    
    return avg_loss, rmse

# --- 6. MAIN EXECUTION SCRIPT ---

if __name__ == '__main__':
    # Set seed for reproducibility
    np.random.seed(CONFIG.SEED)
    torch.manual_seed(CONFIG.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG.SEED)

    # --- Create a Dummy Dataset (replace with competition data) ---
    # This makes the script runnable without an external CSV file.
    # In a real competition, you would load "train.csv" here.
    dummy_data = """
smiles,target_property
CC(C)c1ccccc1,152.2
c1ccccc1,80.1
CCO, -114.1
C=CC=C, -108.9
c1cnccc1,42.3
O=C(O)c1ccccc1,122.4
CN(C)C=O, -61.0
C1CCCCC1, 6.5
*c1ccccc1*, 100.0  
*CC(*)(C)C*, -50.0 
*OC(=O)c1ccc(C(=O)O)cc1*, 350.0 
"""
    df = pd.read_csv(io.StringIO(dummy_data))
    print("--- Loaded Dummy Data ---")
    print(df.head())
    print("-------------------------")

    # --- Pre-computation and Splitting ---
    # In a real scenario, you might save these pre-processed graphs.
    full_dataset = PolymerDataset(df, target_col='target_property')

    # Filter out any SMILES that failed to parse
    valid_indices = [i for i, data in enumerate(full_dataset) if data is not None]
    if len(valid_indices) != len(full_dataset):
        print(f"Warning: Filtered out {len(full_dataset) - len(valid_indices)} invalid SMILES.")
    
    # A cleaner way to handle valid data
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    clean_dataset = PolymerDataset(df_valid, target_col='target_property')
    
    # Split data
    train_size = int((1.0 - CONFIG.TEST_SPLIT_FRACTION) * len(clean_dataset))
    test_size = len(clean_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(clean_dataset, [train_size, test_size])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

    # --- Model Initialization ---
    # Get feature dimensions from the first data object
    first_data_point = clean_dataset.get(0)
    num_atom_features = first_data_point.num_atom_features
    
    model = PolymerGCN(
        num_atom_features=num_atom_features,
        hidden_channels=CONFIG.HIDDEN_CHANNELS,
        num_gcn_layers=CONFIG.NUM_GCN_LAYERS
    ).to(CONFIG.DEVICE)
    
    print("\n--- Model Architecture ---")
    print(model)
    print("------------------------\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss() # Use MSE for RMSE calculation

    # --- Training Loop ---
    best_test_rmse = float('inf')
    for epoch in range(1, CONFIG.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        test_loss, test_rmse = evaluate(model, test_loader, loss_fn)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test RMSE: {test_rmse:.4f}")

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            # In a real competition, you would save the model weights here
            # torch.save(model.state_dict(), 'best_model.pth')
            print(f"  -> New best test RMSE! Saved model checkpoint.")

    print(f"\n--- Training Complete ---")
    print(f"Best Test RMSE Achieved: {best_test_rmse:.4f}")
    print("-------------------------")

