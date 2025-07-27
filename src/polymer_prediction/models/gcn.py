"""GCN-based model for polymer property prediction."""

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Module, Sequential
from torch_geometric.nn import GCNConv, global_mean_pool


class PolymerGCN(Module):
    """Graph Convolutional Network for polymer property prediction."""

    def __init__(self, num_atom_features, hidden_channels, num_gcn_layers):
        """Initialize the GCN model.
        
        Args:
            num_atom_features (int): Number of input features per atom
            hidden_channels (int): Number of hidden channels
            num_gcn_layers (int): Number of GCN layers
        """
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

        # Output layer for multi-target prediction (5 properties)
        self.out = Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            Linear(hidden_channels // 2, 5)  # 5 properties: Tg, FFV, Tc, Density, Rg
        )

    def forward(self, data):
        """Forward pass through the network.
        
        Args:
            data (torch_geometric.data.Data): Input graph data
            
        Returns:
            torch.Tensor: Predicted property values (5 properties)
        """
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
