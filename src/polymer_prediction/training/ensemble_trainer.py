"""Enhanced training with ensemble methods and self-supervised pretraining."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from ..preprocessing.enhanced_featurization import batch_featurize_smiles
from ..utils.competition_metrics import weighted_mae


class EnhancedTrainer:
    """Enhanced trainer with pretraining and ensemble methods."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        target_properties: List[str] = None
    ):
        self.model = model
        self.device = device
        self.target_properties = target_properties or ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Initialize tabular models
        self.tabular_models = {}
        self._init_tabular_models()
    
    def _init_tabular_models(self):
        """Initialize LightGBM models for each property."""
        lgb_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 64,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        for prop in self.target_properties:
            self.tabular_models[prop] = lgb.LGBMRegressor(
                n_estimators=300,
                **lgb_params
            )
    
    def pretrain(
        self,
        pretrain_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 1e-3
    ) -> List[float]:
        """Self-supervised pretraining phase."""
        print("Starting self-supervised pretraining...")
        
        if not hasattr(self.model, 'forward_pretrain'):
            print("Model doesn't support pretraining, skipping...")
            return []
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.model.train()
        pretrain_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch1, batch2 in tqdm(pretrain_loader, desc=f"Pretrain Epoch {epoch+1}"):
                if batch1 is None or batch2 is None:
                    continue
                
                # Move to device
                batch1 = batch1.to(self.device)
                batch2 = batch2.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                emb1 = self.model.forward_pretrain(batch1)
                emb2 = self.model.forward_pretrain(batch2)
                
                # Contrastive loss
                loss = self.model.pretrain_loss(emb1, emb2)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            pretrain_losses.append(avg_loss)
            print(f"Pretrain Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        print("Pretraining completed!")
        return pretrain_losses
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch)
            
            # Masked loss computation
            loss = self._masked_mse_loss(predictions, batch.y, batch.mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
        
        if scheduler is not None:
            scheduler.step()
        
        return total_loss / max(total_samples, 1)
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        return_predictions: bool = False
    ) -> Tuple[float, Dict[str, float], Optional[Tuple]]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        all_masks = []
        
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            batch = batch.to(self.device)
            
            predictions = self.model(batch)
            loss = self._masked_mse_loss(predictions, batch.y, batch.mask)
            
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            
            all_preds.append(predictions.cpu())
            all_targets.append(batch.y.cpu())
            all_masks.append(batch.mask.cpu())
        
        avg_loss = total_loss / max(total_samples, 1)
        
        # Calculate per-property RMSE
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        masks = torch.cat(all_masks, dim=0)
        
        rmses = {}
        for i, prop_name in enumerate(self.target_properties):
            prop_mask = masks[:, i] == 1
            if prop_mask.sum() > 0:
                prop_preds = preds[:, i][prop_mask]
                prop_targets = targets[:, i][prop_mask]
                rmse = torch.sqrt(torch.mean((prop_preds - prop_targets) ** 2))
                rmses[prop_name] = rmse.item()
            else:
                rmses[prop_name] = float('nan')
        
        if return_predictions:
            return avg_loss, rmses, (preds.numpy(), targets.numpy(), masks.numpy())
        else:
            return avg_loss, rmses, None
    
    def train_tabular_ensemble(
        self,
        train_df,
        val_df=None,
        cv_folds: int = 5
    ):
        """Train tabular ensemble models."""
        print("Training tabular ensemble...")
        
        # Extract molecular descriptors
        train_features = batch_featurize_smiles(train_df['SMILES'].tolist())
        
        for prop in self.target_properties:
            if prop not in train_df.columns:
                continue
            
            print(f"Training {prop} tabular model...")
            
            # Get valid samples (non-missing)
            valid_mask = ~train_df[prop].isna()
            if valid_mask.sum() == 0:
                print(f"No valid samples for {prop}, skipping...")
                continue
            
            X_train = train_features[valid_mask]
            y_train = train_df[prop][valid_mask].values
            
            # Cross-validation training
            if cv_folds > 1:
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    # Train fold model
                    fold_model = lgb.LGBMRegressor(**self.tabular_models[prop].get_params())
                    fold_model.fit(X_fold_train, y_fold_train)
                    
                    # Validate
                    y_pred = fold_model.predict(X_fold_val)
                    mae = mean_absolute_error(y_fold_val, y_pred)
                    cv_scores.append(mae)
                
                print(f"{prop} CV MAE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            
            # Train final model on all data
            self.tabular_models[prop].fit(X_train, y_train)
    
    @torch.no_grad()
    def predict_ensemble(
        self,
        test_loader: DataLoader,
        test_df,
        gnn_weight: float = 0.8,
        tabular_weight: float = 0.2
    ) -> Tuple[List[int], np.ndarray]:
        """Generate ensemble predictions."""
        self.model.eval()
        
        # GNN predictions
        gnn_ids = []
        gnn_preds = []
        
        for batch in tqdm(test_loader, desc="GNN Prediction"):
            batch = batch.to(self.device)
            predictions = self.model(batch)
            
            gnn_ids.extend(batch.id)
            gnn_preds.append(predictions.cpu())
        
        gnn_predictions = torch.cat(gnn_preds, dim=0).numpy()
        
        # Tabular predictions
        test_features = batch_featurize_smiles(test_df['SMILES'].tolist())
        tabular_predictions = np.zeros((len(test_df), len(self.target_properties)))
        
        for i, prop in enumerate(self.target_properties):
            if prop in self.tabular_models:
                try:
                    tabular_predictions[:, i] = self.tabular_models[prop].predict(test_features)
                except:
                    print(f"Warning: Tabular prediction failed for {prop}, using GNN only")
                    tabular_predictions[:, i] = gnn_predictions[:, i]
            else:
                tabular_predictions[:, i] = gnn_predictions[:, i]
        
        # Ensemble predictions
        ensemble_predictions = (
            gnn_weight * gnn_predictions + 
            tabular_weight * tabular_predictions
        )
        
        return gnn_ids, ensemble_predictions
    
    def _masked_mse_loss(self, predictions, targets, masks):
        """Calculate MSE loss only for non-missing values."""
        assert predictions.shape == targets.shape == masks.shape
        
        # Only compute loss for non-missing values
        masked_predictions = predictions * masks
        masked_targets = targets * masks
        
        # Calculate squared differences
        squared_diff = (masked_predictions - masked_targets) ** 2
        
        # Sum and normalize
        total_loss = torch.sum(squared_diff)
        total_count = torch.sum(masks)
        
        if total_count > 0:
            return total_loss / total_count
        else:
            return torch.tensor(0.0, device=predictions.device)
    
    def save_checkpoint(self, path: str, epoch: int, optimizer, scheduler, **kwargs):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            **kwargs
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str, optimizer=None, scheduler=None):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint