"""
Example demonstrating the complete stacking ensemble functionality.

This example shows how to use the StackingEnsemble class to combine
GCN and tree ensemble models using cross-validation and meta-learning.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
import sys
import os

# Add src to path
sys.path.append('src')

from polymer_prediction.models.stacking_ensemble import StackingEnsemble


class SimpleGCNModel(nn.Module):
    """Simple GCN model for demonstration."""
    
    def __init__(self, num_atom_features, hidden_channels=64, num_gcn_layers=2):
        super().__init__()
        self.num_atom_features = num_atom_features
        self.hidden_channels = hidden_channels
        
        # Simple feedforward network to simulate GCN
        layers = []
        layers.append(nn.Linear(num_atom_features, hidden_channels))
        layers.append(nn.ReLU())
        
        for _ in range(num_gcn_layers - 1):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_channels, 5))  # 5 target properties
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, data):
        # Simple approach: use global mean pooling with proper gradient handling
        x = data.x  # Node features
        
        # Global mean pooling across all nodes in the batch
        # This maintains gradients properly
        if hasattr(data, 'batch') and data.batch is not None:
            # For batched data, we need to handle each graph separately
            batch_size = int(data.batch.max().item()) + 1
            
            # Create output tensor
            graph_embeddings = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
            
            # Process each graph in the batch
            for i in range(batch_size):
                mask = (data.batch == i)
                if mask.sum() > 0:
                    graph_embeddings[i] = x[mask].mean(dim=0)
                else:
                    # If no nodes, use zeros (shouldn't happen in practice)
                    graph_embeddings[i] = torch.zeros(x.size(1), device=x.device, dtype=x.dtype)
        else:
            # Single graph case
            graph_embeddings = x.mean(dim=0, keepdim=True)
        
        # Pass through the network
        output = self.network(graph_embeddings)
        return output


def create_sample_polymer_data(n_samples=200, n_features=50, random_state=42):
    """Create sample polymer data for demonstration."""
    np.random.seed(random_state)
    
    # Create sample SMILES strings (simplified)
    smiles = [f"C{'C' * (i % 10)}{'O' * ((i // 10) % 3)}" for i in range(n_samples)]
    
    # Create feature matrix (simulating molecular descriptors)
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure to the features
    for i in range(n_features):
        if i % 5 == 0:  # Every 5th feature has some correlation
            X[:, i] += np.sin(np.arange(n_samples) * 0.1)
    
    # Create target properties with realistic relationships
    # Tg (Glass transition temperature)
    tg = 100 + 50 * np.sin(X[:, 0]) + 30 * X[:, 1] + np.random.normal(0, 10, n_samples)
    
    # FFV (Fractional free volume)
    ffv = 0.1 + 0.05 * np.cos(X[:, 2]) + 0.03 * X[:, 3] + np.random.normal(0, 0.02, n_samples)
    
    # Tc (Thermal conductivity)
    tc = 0.2 + 0.1 * X[:, 4] + 0.05 * X[:, 5] + np.random.normal(0, 0.05, n_samples)
    
    # Density
    density = 1.0 + 0.3 * X[:, 6] + 0.2 * X[:, 7] + np.random.normal(0, 0.1, n_samples)
    
    # Rg (Radius of gyration)
    rg = 5.0 + 2.0 * X[:, 8] + 1.5 * X[:, 9] + np.random.normal(0, 0.5, n_samples)
    
    # Add missing values (realistic for experimental data)
    missing_rate = 0.15
    for target in [tg, ffv, tc, density, rg]:
        missing_mask = np.random.random(n_samples) < missing_rate
        target[missing_mask] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': range(n_samples),  # Add id column
        'SMILES': smiles,
        'Tg': tg,
        'FFV': ffv,
        'Tc': tc,
        'Density': density,
        'Rg': rg
    })
    
    return df, X


def create_mock_graph_dataset(df, target_cols, is_test=False):
    """Create mock graph dataset for demonstration."""
    data_list = []
    
    for i, row in df.iterrows():
        # Create dummy molecular graph
        num_nodes = np.random.randint(5, 15)  # Variable number of atoms
        num_features = 10  # Atom features
        
        x = torch.randn(num_nodes, num_features)
        
        # Create edges (random graph structure)
        num_edges = min(num_nodes * 2, num_nodes * (num_nodes - 1) // 2)
        edge_indices = np.random.choice(num_nodes, size=(2, num_edges), replace=True)
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        
        # Create targets and masks
        if not is_test:
            targets = []
            masks = []
            for col in target_cols:
                val = row[col]
                if pd.isna(val):
                    targets.append(0.0)
                    masks.append(0.0)
                else:
                    targets.append(float(val))
                    masks.append(1.0)
            
            # Shape should be (1, 5) to match the real dataset
            y = torch.tensor(targets, dtype=torch.float).unsqueeze(0)
            mask = torch.tensor(masks, dtype=torch.float).unsqueeze(0)
        else:
            # Shape should be (1, 5) to match the real dataset
            y = torch.zeros(len(target_cols), dtype=torch.float).unsqueeze(0)
            mask = torch.zeros(len(target_cols), dtype=torch.float).unsqueeze(0)
        
        data = Data(x=x, edge_index=edge_index, y=y, mask=mask)
        data.id = i  # Store as integer
        data_list.append(data)
    
    return data_list


class MockPolymerDataset:
    """Mock dataset for demonstration."""
    
    def __init__(self, df, target_cols, is_test=False):
        self.df = df
        self.target_cols = target_cols
        self.is_test = is_test
        self.data_list = create_mock_graph_dataset(df, target_cols, is_test)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


def demonstrate_stacking_ensemble():
    """Demonstrate the complete stacking ensemble workflow."""
    print("Stacking Ensemble Demonstration")
    print("=" * 50)
    
    # 1. Create sample data
    print("1. Creating sample polymer data...")
    df, X = create_sample_polymer_data(n_samples=150, n_features=30)
    
    print(f"   - Dataset size: {len(df)} samples")
    print(f"   - Feature dimensions: {X.shape[1]}")
    print(f"   - Target properties: {['Tg', 'FFV', 'Tc', 'Density', 'Rg']}")
    
    # Check missing values
    missing_counts = df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].isna().sum()
    print(f"   - Missing values per target: {dict(missing_counts)}")
    
    # 2. Split data
    print("\n2. Splitting data into train/test...")
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    X_train = X[:train_size]
    X_test = X[train_size:]
    
    print(f"   - Training samples: {len(train_df)}")
    print(f"   - Test samples: {len(test_df)}")
    
    # 3. Mock the dataset (to avoid complex GCN setup)
    print("\n3. Setting up mock dataset...")
    import polymer_prediction.data.dataset as dataset_module
    original_dataset = dataset_module.PolymerDataset
    dataset_module.PolymerDataset = MockPolymerDataset
    
    try:
        # 4. Create and configure stacking ensemble
        print("\n4. Creating stacking ensemble...")
        ensemble = StackingEnsemble(
            gcn_model_class=SimpleGCNModel,
            gcn_params={
                'hidden_channels': 64,
                'num_gcn_layers': 3
            },
            tree_models=['lgbm', 'xgb'],  # Use LightGBM and XGBoost
            cv_folds=5,
            gcn_epochs=20,
            optimize_tree_hyperparams=False,  # Skip for demo speed
            random_state=42,
            device=torch.device('cpu'),
            batch_size=32
        )
        
        print("   - GCN model: SimpleGCNModel with 64 hidden channels")
        print("   - Tree models: LightGBM, XGBoost")
        print("   - Cross-validation: 5 folds")
        print("   - Meta-learner: Ridge regression")
        
        # 5. Train the ensemble
        print("\n5. Training stacking ensemble...")
        print("   This may take a few minutes...")
        
        # Prepare target matrix
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        y_train = train_df[target_cols].values
        
        ensemble.fit(train_df, X_train, y_train)
        
        print("   ✓ Training completed!")
        
        # 6. Get training information
        print("\n6. Training results:")
        cv_scores = ensemble.get_cv_scores()
        model_info = ensemble.get_model_info()
        
        print(f"   - Base models trained: {model_info['base_models']}")
        print(f"   - Meta-models trained: {len(model_info['meta_models'])} targets")
        
        if 'gcn' in cv_scores:
            print("   - GCN CV scores:")
            for prop, score in cv_scores['gcn'].items():
                if not np.isnan(score):
                    print(f"     {prop}: {score:.4f} RMSE")
        
        if 'tree' in cv_scores:
            print("   - Tree ensemble CV scores:")
            for prop, score in cv_scores['tree'].items():
                if not np.isnan(score):
                    print(f"     {prop}: {score:.4f} RMSE")
        
        # 7. Make predictions
        print("\n7. Making predictions on test set...")
        
        # Clear test targets for prediction
        test_df_pred = test_df.copy()
        for col in target_cols:
            test_df_pred[col] = np.nan
        
        predictions = ensemble.predict(test_df_pred, X_test)
        
        print(f"   - Prediction shape: {predictions.shape}")
        print(f"   - Valid predictions: {np.sum(~np.isnan(predictions))}/{predictions.size}")
        
        # 8. Evaluate predictions (where we have ground truth)
        print("\n8. Evaluation on test set:")
        
        y_test = test_df[target_cols].values
        
        for i, prop in enumerate(target_cols):
            # Find samples where both prediction and ground truth are available
            mask = ~np.isnan(y_test[:, i]) & ~np.isnan(predictions[:, i])
            
            if np.sum(mask) > 0:
                y_true = y_test[mask, i]
                y_pred = predictions[mask, i]
                
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                mae = np.mean(np.abs(y_true - y_pred))
                
                print(f"   {prop}:")
                print(f"     - Samples: {np.sum(mask)}")
                print(f"     - RMSE: {rmse:.4f}")
                print(f"     - MAE: {mae:.4f}")
        
        # 9. Show sample predictions
        print("\n9. Sample predictions:")
        print("   (First 5 test samples)")
        
        for i in range(min(5, len(predictions))):
            print(f"   Sample {i + 1}:")
            for j, prop in enumerate(target_cols):
                pred_val = predictions[i, j]
                true_val = y_test[i, j]
                
                if not np.isnan(pred_val):
                    if not np.isnan(true_val):
                        print(f"     {prop}: {pred_val:.3f} (true: {true_val:.3f})")
                    else:
                        print(f"     {prop}: {pred_val:.3f} (true: missing)")
                else:
                    print(f"     {prop}: missing")
        
        print("\n✅ Stacking ensemble demonstration completed successfully!")
        
        print("\nKey features demonstrated:")
        print("- Cross-validation framework for out-of-fold predictions")
        print("- Meta-learner training using Ridge regression")
        print("- Combination of GCN and tree ensemble models")
        print("- Proper handling of missing values")
        print("- End-to-end prediction pipeline")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Restore original dataset
        dataset_module.PolymerDataset = original_dataset


if __name__ == "__main__":
    demonstrate_stacking_ensemble()