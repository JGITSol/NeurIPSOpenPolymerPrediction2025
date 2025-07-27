#!/usr/bin/env python3
"""Debug script to check tensor shapes"""

import sys
sys.path.append('src')

import pandas as pd
from polymer_prediction.data.dataset import PolymerDataset
from torch_geometric.data import DataLoader

# Load a small sample of data
train_df = pd.read_csv("info/train.csv").head(10)
print("Data shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())

# Create dataset
dataset = PolymerDataset(train_df, is_test=False)
print("Dataset length:", len(dataset))

# Check a single sample
sample = dataset.get(0)
if sample is not None:
    print("Sample y shape:", sample.y.shape)
    print("Sample mask shape:", sample.mask.shape)
    print("Sample y:", sample.y)
    print("Sample mask:", sample.mask)
    print("Sample x shape:", sample.x.shape)
    print("Sample edge_index shape:", sample.edge_index.shape)
else:
    print("Sample is None")

# Check batch
loader = DataLoader(dataset, batch_size=4, shuffle=False)
for batch in loader:
    print("Batch y shape:", batch.y.shape)
    print("Batch mask shape:", batch.mask.shape)
    print("Batch num_graphs:", batch.num_graphs)
    break