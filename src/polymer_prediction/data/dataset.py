"""Dataset classes for polymer prediction."""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Dataset

from polymer_prediction.preprocessing.featurization import smiles_to_graph


class PolymerDataset(Dataset):
    """PyTorch Geometric Dataset for polymer data.
    
    Handles on-the-fly conversion of SMILES to graphs and multi-target prediction.
    """

    def __init__(self, df, target_cols=None, is_test=False):
        """Initialize the dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame containing SMILES and target values
            target_cols (list): List of target column names. If None, uses competition targets.
            is_test (bool): Whether this is test data (no targets)
        """
        super().__init__()
        self.df = df
        self.is_test = is_test
        
        # Default target columns for the competition
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
        """Return the number of samples in the dataset."""
        return len(self.df)

    def get(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            torch_geometric.data.Data: Graph representation of the molecule
        """
        if idx in self.cache:
            data = self.cache[idx]
        else:
            smiles = self.smiles_list[idx]
            data = smiles_to_graph(smiles)
            if data is None:  # Handle RDKit parsing errors
                return None
            self.cache[idx] = data

        # Add ID (store as integer, not tensor)
        data.id = int(self.ids[idx])
        
        if not self.is_test:
            # Add target values and masks - reshape to (1, 5) to preserve structure during batching
            data.y = torch.tensor(self.targets[idx], dtype=torch.float).unsqueeze(0)  # Shape: (1, 5)
            data.mask = torch.tensor(self.masks[idx], dtype=torch.float).unsqueeze(0)  # Shape: (1, 5)
        
        return data
