"""Training pipeline for polymer prediction models."""

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

from polymer_prediction.config.config import Config
from polymer_prediction.data.dataset import PolymerDataset
from polymer_prediction.models.gcn import PolymerGCN
from polymer_prediction.models.ensemble import TreeEnsemble
from polymer_prediction.models.stacking_ensemble import StackingEnsemble
from polymer_prediction.training.trainer import train_one_epoch, evaluate, predict, masked_mse_loss
from polymer_prediction.utils.logging import get_logger
from polymer_prediction.utils.path_manager import PathManager
from polymer_prediction.utils.error_handling import ErrorHandler, MemoryManager, DeviceManager

logger = get_logger(__name__)


class TrainingPipeline:
    """Training pipeline for polymer prediction models."""
    
    def __init__(self, config: Config, path_manager: Optional[PathManager] = None):
        """Initialize training pipeline.
        
        Args:
            config: Configuration object
            path_manager: Path manager for file operations
        """
        self.config = config
        self.path_manager = path_manager or PathManager()
        self.error_handler = ErrorHandler()
        self.memory_manager = MemoryManager(self.error_handler)
        self.device_manager = DeviceManager(self.error_handler)
        
        # Training state
        self.models = {}
        self.training_history = {}
        
        logger.info("TrainingPipeline initialized")
    
    def create_data_loaders(self, train_dataset: PolymerDataset, 
                           val_dataset: Optional[PolymerDataset] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create data loaders for training.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Creating data loaders...")
        
        # Determine batch size based on available memory
        batch_size = self.memory_manager.get_optimal_batch_size(
            self.config.training.batch_size,
            self.error_handler.get_memory_info().get('system_available_gb', 4.0)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory and self.config.device.type == 'cuda',
            drop_last=self.config.data.drop_last
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers,
                pin_memory=self.config.data.pin_memory and self.config.device.type == 'cuda',
                drop_last=False
            )
        
        logger.info(f"Data loaders created with batch size: {batch_size}")
        
        return train_loader, val_loader
    
    def initialize_gcn_model(self, num_atom_features: int) -> PolymerGCN:
        """Initialize GCN model.
        
        Args:
            num_atom_features: Number of atom features
            
        Returns:
            Initialized GCN model
        """
        logger.info("Initializing GCN model...")
        
        model = PolymerGCN(
            num_atom_features=num_atom_features,
            hidden_channels=self.config.model.hidden_channels,
            num_gcn_layers=self.config.model.num_gcn_layers
        )
        
        # Move model to device
        model = self.device_manager.safe_device_transfer(model, self.config.device)
        
        logger.info(f"GCN model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def train_gcn_model(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Train GCN model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting GCN model training...")
        
        # Get sample data to determine feature count
        sample_data = next(iter(train_loader))
        num_atom_features = sample_data.x.size(1)
        
        # Initialize model
        model = self.initialize_gcn_model(num_atom_features)
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        scheduler = None
        if self.config.training.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.training.num_epochs
            )
        
        # Training loop
        training_history = {
            "train_losses": [],
            "val_losses": [],
            "learning_rates": []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.training.num_epochs):
            # Training phase
            train_loss = train_one_epoch(model, train_loader, optimizer, self.config.device)
            training_history["train_losses"].append(train_loss)
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                val_loss = evaluate(model, val_loader, self.config.device)
                training_history["val_losses"].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model(model, "best_gcn_model.pt")
                else:
                    patience_counter += 1
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
                training_history["learning_rates"].append(optimizer.param_groups[0]['lr'])
            
            # Logging
            if epoch % 10 == 0:
                log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss = {val_loss:.4f}"
                logger.info(log_msg)
            
            # Early stopping
            if patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Memory cleanup
            if epoch % self.config.performance.memory_cleanup_frequency == 0:
                self.error_handler.force_garbage_collection()
        
        # Store model and history
        self.models["gcn"] = model
        self.training_history["gcn"] = training_history
        
        results = {
            "model": model,
            "training_history": training_history,
            "best_val_loss": best_val_loss,
            "epochs_completed": epoch + 1
        }
        
        logger.info("GCN model training completed")
        
        return results
    
    def train_tree_ensemble(self, train_dataset: PolymerDataset) -> Dict[str, Any]:
        """Train tree ensemble models.
        
        Args:
            train_dataset: Training dataset
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting tree ensemble training...")
        
        # Extract features and targets from dataset
        features = []
        targets = []
        
        for data in train_dataset:
            if data is not None:
                # Use molecular descriptors as features for tree models
                # This is a simplified approach - in practice, you'd extract RDKit descriptors
                feature_vector = data.x.mean(dim=0).numpy()  # Simple aggregation
                features.append(feature_vector)
                
                if hasattr(data, 'y') and data.y is not None:
                    targets.append(data.y.squeeze().numpy())
        
        if not features:
            raise ValueError("No valid features extracted for tree ensemble training")
        
        features = np.array(features)
        targets = np.array(targets)
        
        logger.info(f"Extracted features: {features.shape}, targets: {targets.shape}")
        
        # Initialize and train tree ensemble
        tree_ensemble = TreeEnsemble(models=self.config.model.tree_models)
        tree_ensemble.fit(features, targets)
        
        # Store model
        self.models["tree_ensemble"] = tree_ensemble
        
        results = {
            "model": tree_ensemble,
            "feature_shape": features.shape,
            "target_shape": targets.shape
        }
        
        logger.info("Tree ensemble training completed")
        
        return results
    
    def train_stacking_ensemble(self, train_dataset: PolymerDataset) -> Dict[str, Any]:
        """Train stacking ensemble combining GCN and tree models.
        
        Args:
            train_dataset: Training dataset
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting stacking ensemble training...")
        
        if "gcn" not in self.models or "tree_ensemble" not in self.models:
            raise ValueError("Both GCN and tree ensemble models must be trained before stacking")
        
        # Create stacking ensemble
        from polymer_prediction.models.gcn import PolymerGCN
        
        stacking_ensemble = StackingEnsemble(
            gcn_model_class=PolymerGCN,
            gcn_params={
                'num_atom_features': 26,  # Standard RDKit features
                'hidden_channels': self.config.model.hidden_channels,
                'num_gcn_layers': self.config.model.num_gcn_layers
            },
            tree_models=self.config.model.tree_models,
            cv_folds=self.config.model.n_folds,
            device=self.config.device,
            batch_size=self.config.training.batch_size,
            gcn_epochs=min(10, self.config.training.num_epochs)  # Reduced epochs for CV
        )
        
        # Extract features and targets for stacking ensemble
        features = []
        targets = []
        ids = []
        
        for data in train_dataset:
            if data is not None:
                # Extract features (same as tree ensemble)
                feature_vector = data.x.mean(dim=0).numpy()  # Simple aggregation
                features.append(feature_vector)
                
                if hasattr(data, 'y') and data.y is not None:
                    targets.append(data.y.squeeze().numpy())
                
                if hasattr(data, 'id'):
                    ids.append(data.id)
        
        if not features:
            raise ValueError("No valid features extracted for stacking ensemble training")
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Create DataFrame for stacking ensemble
        df = pd.DataFrame({
            'id': ids if ids else range(len(features)),
            'SMILES': ['dummy'] * len(features)  # Placeholder SMILES
        })
        
        # Add target columns to DataFrame
        for i, col in enumerate(self.config.data.target_cols):
            if i < targets.shape[1]:
                df[col] = targets[:, i]
        
        # Train stacking ensemble
        stacking_ensemble.fit(df, features, targets)
        
        # Store model
        self.models["stacking_ensemble"] = stacking_ensemble
        
        results = {
            "model": stacking_ensemble,
            "n_folds": self.config.model.n_folds,
            "meta_model": self.config.model.meta_model
        }
        
        logger.info("Stacking ensemble training completed")
        
        return results
    
    def save_model(self, model: nn.Module, filename: str) -> Path:
        """Save a model to disk.
        
        Args:
            model: Model to save
            filename: Filename for the saved model
            
        Returns:
            Path to the saved model file
        """
        model_path = self.path_manager.get_model_path(filename)
        
        try:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
        
        return model_path
    
    def load_model(self, model_class: type, filename: str, **model_kwargs) -> nn.Module:
        """Load a model from disk.
        
        Args:
            model_class: Model class to instantiate
            filename: Filename of the saved model
            **model_kwargs: Keyword arguments for model initialization
            
        Returns:
            Loaded model
        """
        model_path = self.path_manager.get_model_path(filename)
        
        if not self.path_manager.file_exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model = model_class(**model_kwargs)
            model.load_state_dict(torch.load(model_path, map_location=self.config.device))
            model = self.device_manager.safe_device_transfer(model, self.config.device)
            
            logger.info(f"Model loaded from {model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def save_training_history(self, filename: str = "training_history.json") -> Path:
        """Save training history to disk.
        
        Args:
            filename: Filename for the training history
            
        Returns:
            Path to the saved history file
        """
        history_path = self.path_manager.get_output_path(filename)
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for model_name, history in self.training_history.items():
                serializable_history[model_name] = {}
                for key, value in history.items():
                    if isinstance(value, np.ndarray):
                        serializable_history[model_name][key] = value.tolist()
                    elif isinstance(value, list):
                        serializable_history[model_name][key] = value
                    else:
                        serializable_history[model_name][key] = str(value)
            
            with open(history_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
            
            logger.info(f"Training history saved to {history_path}")
            
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
            raise
        
        return history_path
    
    def run_training_pipeline(self, train_dataset: PolymerDataset, 
                             val_dataset: Optional[PolymerDataset] = None) -> Dict[str, Any]:
        """Run the complete training pipeline.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Training results dictionary
        """
        logger.info("Running training pipeline...")
        
        results = {}
        
        try:
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(train_dataset, val_dataset)
            
            # Train GCN model
            gcn_results = self.train_gcn_model(train_loader, val_loader)
            results["gcn"] = gcn_results
            
            # Train tree ensemble
            tree_results = self.train_tree_ensemble(train_dataset)
            results["tree_ensemble"] = tree_results
            
            # Train stacking ensemble if enabled
            if self.config.model.use_stacking:
                stacking_results = self.train_stacking_ensemble(train_dataset)
                results["stacking_ensemble"] = stacking_results
            
            # Save training history
            self.save_training_history()
            
            logger.info("Training pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
        
        return results