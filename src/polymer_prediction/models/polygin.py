"""PolyGIN: Enhanced Graph Isomorphism Network for polymer property prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.norm import BatchNorm
import torch_geometric.transforms as T
from typing import Optional


class PolyGIN(nn.Module):
    """Enhanced Graph Isomorphism Network with virtual node for polymer prediction."""
    
    def __init__(
        self,
        num_atom_features: int,
        num_bond_features: int = 0,
        hidden_channels: int = 96,
        num_layers: int = 8,
        dropout: float = 0.15,
        num_targets: int = 5,
        use_virtual_node: bool = True,
        pooling: str = 'mean'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_virtual_node = use_virtual_node
        self.pooling = pooling
        
        # Input projection
        self.atom_encoder = nn.Linear(num_atom_features, hidden_channels)
        
        # Virtual node for global context
        if use_virtual_node:
            self.virtual_node_emb = nn.Embedding(1, hidden_channels)
            self.virtual_node_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
        
        # GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.SiLU(),
                nn.Linear(2 * hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp, eps=0.0, train_eps=False))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Readout layers
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        elif pooling == 'concat':
            self.pool = lambda x, batch: torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch)
            ], dim=1)
            hidden_channels *= 3
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.BatchNorm1d(hidden_channels // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, num_targets)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, data):
        """Forward pass through the network."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encode atom features
        x = self.atom_encoder(x)
        
        # Add virtual node if enabled
        if self.use_virtual_node:
            # Get number of graphs in batch
            num_graphs = batch.max().item() + 1
            
            # Create virtual node embeddings
            virtual_node_feat = self.virtual_node_emb(torch.zeros(num_graphs, dtype=torch.long, device=x.device))
            
            # Add virtual nodes to the graph
            virtual_node_idx = torch.arange(num_graphs, device=x.device) + x.size(0)
            
            # Connect virtual node to all nodes in each graph
            virtual_edges = []
            for i in range(num_graphs):
                graph_nodes = (batch == i).nonzero(as_tuple=True)[0]
                virtual_node = virtual_node_idx[i]
                
                # Bidirectional connections
                for node in graph_nodes:
                    virtual_edges.extend([[virtual_node, node], [node, virtual_node]])
            
            if virtual_edges:
                virtual_edge_index = torch.tensor(virtual_edges, dtype=torch.long, device=x.device).t()
                edge_index = torch.cat([edge_index, virtual_edge_index], dim=1)
            
            # Concatenate virtual node features
            x = torch.cat([x, virtual_node_feat], dim=0)
            
            # Update batch indices
            virtual_batch = torch.arange(num_graphs, device=batch.device)
            batch = torch.cat([batch, virtual_batch], dim=0)
        
        # Message passing through GIN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.silu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if i > 0:
                x = x + x_new
            else:
                x = x_new
            
            # Update virtual node features
            if self.use_virtual_node and i < self.num_layers - 1:
                # Extract virtual node features
                virtual_feats = x[-num_graphs:]
                # Update virtual node features
                virtual_feats = self.virtual_node_mlp(virtual_feats)
                # Replace virtual node features
                x[-num_graphs:] = virtual_feats
        
        # Remove virtual nodes before pooling
        if self.use_virtual_node:
            x = x[:-num_graphs]
            batch = batch[:-num_graphs]
        
        # Graph-level pooling
        x = self.pool(x, batch)
        
        # Final prediction
        return self.head(x)


class PolyGINWithPretraining(PolyGIN):
    """PolyGIN with self-supervised pretraining capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Pretraining head for self-supervised learning
        hidden_channels = kwargs.get('hidden_channels', 96)
        self.pretrain_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
    
    def forward_pretrain(self, data):
        """Forward pass for pretraining (returns embeddings)."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encode atom features
        x = self.atom_encoder(x)
        
        # Message passing (simplified for pretraining)
        for i, (conv, bn) in enumerate(zip(self.convs[:4], self.batch_norms[:4])):  # Use fewer layers
            x = conv(x, edge_index)
            x = bn(x)
            x = F.silu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        x = self.pool(x, batch)
        
        # Return embeddings for contrastive learning
        return self.pretrain_head(x)
    
    def pretrain_loss(self, embeddings1, embeddings2, temperature=0.1):
        """Contrastive loss for self-supervised pretraining."""
        # Normalize embeddings
        embeddings1 = F.normalize(embeddings1, dim=1)
        embeddings2 = F.normalize(embeddings2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings1, embeddings2.t()) / temperature
        
        # Labels for contrastive learning (diagonal elements are positive pairs)
        labels = torch.arange(embeddings1.size(0), device=embeddings1.device)
        
        # Contrastive loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss