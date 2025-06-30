"""Dataset classes for polymer prediction."""

import torch
from torch_geometric.data import Dataset

from polymer_prediction.preprocessing.featurization import smiles_to_graph


class PolymerDataset(Dataset):
    """PyTorch Geometric Dataset for polymer data.
    
    Handles on-the-fly conversion of SMILES to graphs.
    """

    def __init__(self, df, target_col):
        """Initialize the dataset.
        
        Args:
            df (pandas.DataFrame): DataFrame containing SMILES and target values
            target_col (str): Name of the column containing target values
        """
        super().__init__()
        self.df = df
        self.smiles_list = df['smiles'].tolist()
        self.targets = df[target_col].tolist()
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

        # Add target value to the data object
        data.y = torch.tensor([self.targets[idx]], dtype=torch.float)
        return data
