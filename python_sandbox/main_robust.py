"""
Robust Polymer Prediction Pipeline with Performance Optimization

This module demonstrates the integration of comprehensive error handling,
robustness features, and performance optimization for the polymer prediction system.
"""

import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch_geometric.data
from tqdm import tqdm
import gc
from pathlib import Path

# Import existing production components
sys.path.append('src')
from polymer_prediction.data.dataset import PolymerDataset
from polymer_prediction.training.trainer import (
    masked_mse_loss, 
    train_one_epoch, 
    evaluate, 
    predict
)
from polymer_prediction.preprocessing.featurization import smiles_to_graph
from polymer_prediction.config.config import CONFIG

# Import enhanced error handling
from polymer_prediction.utils.error_handling import (
    ErrorHandler,
    SMILESValidator,
    MemoryManager,
    DeviceManager,
    InputValidator,
    robust_function_wrapper,
    global_error_handler
)
from polymer_prediction.utils.logging import setup_logging, get_logger

# Import performance optimization components
from polymer_prediction.optimization import (
    create_optimized_pipeline,
    run_optimized_training,
    PerformanceConfig
)
from polymer_prediction.utils.performance import (
    PerformanceOptimizer,
    GraphCache,
    CPUOptimizer,
    MemoryMonitor,
    ProgressTracker
)
from polymer_prediction.data.optimized_dataloader import create_optimized_dataloader
from polymer_prediction.training.optimized_trainer import create_optimized_trainer

# Setup enhanced logging
setup_logging(log_level="INFO", log_file="logs/polymer_prediction_robust.log")
logger = get_logger(__name__)


class RobustConfig:
    """Enhanced configuration class with error handling integration."""
    
    def __init__(self):
        # Initialize error handling components
        self.error_handler = ErrorHandler()
        self.smiles_validator = SMILESValidator(self.error_handler)
        self.memory_manager = MemoryManager(self.error_handler)
        self.device_manager = DeviceManager(self.error_handler)
        self.input_validator = InputValidator(self.error_handler)
        
        # Detect optimal device
        self.DEVICE = self.device_manager.detect_optimal_device()
        
        # Base configuration from existing CONFIG
        try:
            self.BATCH_SIZE = CONFIG.BATCH_SIZE
            self.LEARNING_RATE = CONFIG.LEARNING_RATE
            self.HIDDEN_CHANNELS = CONFIG.HIDDEN_CHANNELS
            self.NUM_GCN_LAYERS = CONFIG.NUM_GCN_LAYERS
            self.NUM_EPOCHS = CONFIG.NUM_EPOCHS
        except Exception as e:
            logger.warning(f"Could not load existing CONFIG: {e}. Using defaults.")
            self.BATCH_SIZE = 32
            self.LEARNING_RATE = 1e-3
            self.HIDDEN_CHANNELS = 128
            self.NUM_GCN_LAYERS = 3
            self.NUM_EPOCHS = 50
        
        # Additional robust configuration
        self.DATA_PATH = 'info'
        self.TARGET_COLS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.N_FOLDS = 5
        self.RANDOM_STATE = 42
        self.MAX_RETRIES = 3
        self.MIN_BATCH_SIZE = 1
        
        # Optimize batch size based on available memory
        memory_info = self.error_handler.get_memory_info()
        available_memory = memory_info.get('system_available_gb', 4.0)
        self.BATCH_SIZE = self.memory_manager.get_optimal_batch_size(
            self.BATCH_SIZE, available_memory
        )
        
        # CPU optimizations
        if self.DEVICE.type == 'cpu':
            self.BATCH_SIZE = min(self.BATCH_SIZE, 16)
            self.NUM_EPOCHS = min(self.NUM_EPOCHS, 50)
        
        logger.info(f"Robust configuration initialized: Device={self.DEVICE}, Batch size={self.BATCH_SIZE}")


config = RobustConfig()
np.random.seed(config.RANDOM_STATE)
torch.manual_seed(config.RANDOM_STATE)


@robust_function_wrapper(config.error_handler, fallback_return=(None, None))
def load_data_robust():
    """Load training and test data with comprehensive error handling."""
    logger.info("Loading data with robust error handling...")
    
    # Validate file paths
    train_path = config.input_validator.validate_file_path(f'{config.DATA_PATH}/train.csv')
    test_path = config.input_validator.validate_file_path(f'{config.DATA_PATH}/test.csv')
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Validate DataFrame structure
    required_train_cols = ['id', 'SMILES'] + config.TARGET_COLS
    required_test_cols = ['id', 'SMILES']
    
    train_df = config.input_validator.validate_dataframe(
        train_df, required_train_cols, "Training DataFrame"
    )
    test_df = config.input_validator.validate_dataframe(
        test_df, required_test_cols, "Test DataFrame"
    )
    
    # Validate and filter SMILES
    logger.info("Validating SMILES strings...")
    train_df = config.smiles_validator.filter_valid_smiles(train_df, 'SMILES')
    test_df = config.smiles_validator.filter_valid_smiles(test_df, 'SMILES')
    
    # Validate target columns
    valid_targets = config.input_validator.validate_target_columns(train_df, config.TARGET_COLS)
    config.TARGET_COLS = valid_targets
    
    logger.info(f"Data loaded successfully: Train {train_df.shape}, Test {test_df.shape}")
    return train_df, test_df


class RobustPolymerDataset(PolymerDataset):
    """Enhanced PolymerDataset with robust error handling."""
    
    def __init__(self, df, target_cols=None, is_test=False, error_handler=None):
        """Initialize robust dataset.
        
        Args:
            df: DataFrame containing SMILES and target values
            target_cols: List of target column names
            is_test: Whether this is test data
            error_handler: Error handler instance
        """
        self.error_handler = error_handler or global_error_handler
        super().__init__(df, target_cols, is_test)
        
        # Pre-validate all SMILES
        self.valid_indices = []
        for idx, smiles in enumerate(self.smiles_list):
            if self._validate_smiles_safe(smiles, idx):
                self.valid_indices.append(idx)
        
        logger.info(f"Dataset initialized: {len(self.valid_indices)}/{len(self.smiles_list)} valid samples")
    
    def _validate_smiles_safe(self, smiles: str, index: int) -> bool:
        """Safely validate SMILES with error handling."""
        try:
            mol = smiles_to_graph(smiles)
            if mol is None:
                self.error_handler.handle_invalid_smiles(smiles, index, "Graph conversion failed")
                return False
            return True
        except Exception as e:
            self.error_handler.handle_invalid_smiles(smiles, index, f"Exception: {str(e)}")
            return False
    
    def len(self):
        """Return number of valid samples."""
        return len(self.valid_indices)
    
    def get(self, idx):
        """Get a sample with robust error handling."""
        if idx >= len(self.valid_indices):
            return None
        
        actual_idx = self.valid_indices[idx]
        
        try:
            # Use parent's get method but with actual index
            original_len = len(self.smiles_list)
            
            # Temporarily adjust for parent class
            if actual_idx < original_len:
                smiles = self.smiles_list[actual_idx]
                data = smiles_to_graph(smiles)
                
                if data is None:
                    return None
                
                # Add ID
                data.id = int(self.ids[actual_idx])
                
                if not self.is_test:
                    # Add target values and masks
                    target_values = []
                    mask_values = []
                    
                    for col in self.target_cols:
                        if col in self.df.columns:
                            val = self.df.iloc[actual_idx][col]
                            if pd.isna(val):
                                target_values.append(0.0)
                                mask_values.append(0.0)
                            else:
                                target_values.append(float(val))
                                mask_values.append(1.0)
                        else:
                            target_values.append(0.0)
                            mask_values.append(0.0)
                    
                    data.y = torch.tensor(target_values, dtype=torch.float).unsqueeze(0)
                    data.mask = torch.tensor(mask_values, dtype=torch.float).unsqueeze(0)
                
                return data
            
        except Exception as e:
            logger.warning(f"Error getting sample {idx} (actual {actual_idx}): {str(e)}")
            return None
        
        return None


def create_robust_dataloader(dataset, batch_size, shuffle=False, error_handler=None):
    """Create DataLoader with robust error handling and memory management."""
    error_handler = error_handler or global_error_handler
    
    def safe_collate_fn(batch):
        """Collate function that handles None values and memory issues."""
        # Filter out None values
        valid_batch = [item for item in batch if item is not None]
        
        if len(valid_batch) == 0:
            return None
        
        try:
            return torch_geometric.data.Batch.from_data_list(valid_batch)
        except Exception as e:
            logger.warning(f"Batch collation failed: {str(e)}")
            return None
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=safe_collate_fn,
        drop_last=False,
        num_workers=0  # Avoid multiprocessing issues
    )


class RobustPolymerGCN(nn.Module):
    """Enhanced GCN model with robust error handling."""
    
    def __init__(self, num_atom_features, hidden_channels=None, num_gcn_layers=None, device_manager=None):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        self.device_manager = device_manager or config.device_manager
        
        # Use config values
        hidden_channels = hidden_channels or config.HIDDEN_CHANNELS
        num_gcn_layers = num_gcn_layers or config.NUM_GCN_LAYERS
        
        self.convs = nn.ModuleList([GCNConv(num_atom_features, hidden_channels)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        
        for _ in range(num_gcn_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_channels, len(config.TARGET_COLS))
        self.global_mean_pool = global_mean_pool
    
    def forward(self, data):
        """Forward pass with error handling."""
        try:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, edge_index)))
                x = self.dropout(x)
            
            x = self.global_mean_pool(x, batch)
            return self.out(x)
            
        except Exception as e:
            logger.error(f"Error in GCN forward pass: {str(e)}")
            # Return dummy output to prevent complete failure
            batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
            return torch.zeros(batch_size, len(config.TARGET_COLS), device=data.x.device)


def train_gcn_robust(train_df, test_df):
    """Train GCN with comprehensive error handling and robustness features."""
    logger.info("Starting robust GCN training...")
    
    current_batch_size = config.BATCH_SIZE
    retry_count = 0
    
    while retry_count < config.MAX_RETRIES:
        try:
            with config.memory_manager.memory_cleanup_context():
                # Create robust datasets
                dataset = RobustPolymerDataset(
                    train_df, 
                    target_cols=config.TARGET_COLS, 
                    is_test=False,
                    error_handler=config.error_handler
                )
                test_dataset = RobustPolymerDataset(
                    test_df, 
                    target_cols=config.TARGET_COLS, 
                    is_test=True,
                    error_handler=config.error_handler
                )
                
                # Create robust DataLoaders
                loader = create_robust_dataloader(
                    dataset, current_batch_size, shuffle=True, error_handler=config.error_handler
                )
                test_loader = create_robust_dataloader(
                    test_dataset, current_batch_size, shuffle=False, error_handler=config.error_handler
                )
                
                # Get sample for feature count
                sample_data = None
                for data in dataset:
                    if data is not None:
                        sample_data = data
                        break
                
                if sample_data is None:
                    raise ValueError("No valid molecular graphs found in dataset")
                
                num_atom_features = sample_data.x.size(1)
                logger.info(f"Number of atom features: {num_atom_features}")
                
                # Initialize robust model
                model = RobustPolymerGCN(
                    num_atom_features=num_atom_features,
                    device_manager=config.device_manager
                )
                
                # Safe device transfer
                model = config.device_manager.safe_device_transfer(model)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
                
                logger.info(f"Starting GCN training with batch size {current_batch_size}...")
                
                # Training loop with memory monitoring
                for epoch in range(config.NUM_EPOCHS):
                    try:
                        # Check memory before each epoch
                        if not config.memory_manager.check_memory_usage():
                            logger.warning("High memory usage detected, forcing cleanup")
                            config.error_handler.force_garbage_collection()
                        
                        # Train one epoch with error handling
                        avg_loss = train_one_epoch_robust(model, loader, optimizer, config.DEVICE)
                        
                        if epoch % 10 == 0:
                            logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
                            
                            # Memory cleanup every 10 epochs
                            config.error_handler.force_garbage_collection()
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning(f"Memory error during training at epoch {epoch}")
                            current_batch_size = config.memory_manager.adaptive_batch_size_reduction(
                                current_batch_size, f"training epoch {epoch}"
                            )
                            
                            if current_batch_size < config.MIN_BATCH_SIZE:
                                raise MemoryError("Batch size reduced below minimum threshold")
                            
                            # Restart training with smaller batch size
                            raise e
                        else:
                            raise e
                
                # Generate test predictions
                logger.info("Generating robust test predictions...")
                test_ids, test_preds = predict_robust(model, test_loader, config.DEVICE)
                
                logger.info("Robust GCN training completed successfully!")
                return test_preds
                
        except Exception as e:
            retry_count += 1
            logger.error(f"Training attempt {retry_count} failed: {str(e)}")
            
            if retry_count < config.MAX_RETRIES:
                # Try with reduced batch size
                current_batch_size = config.memory_manager.adaptive_batch_size_reduction(
                    current_batch_size, f"training retry {retry_count}"
                )
                
                if current_batch_size < config.MIN_BATCH_SIZE:
                    logger.error("Batch size reduced below minimum threshold. Aborting.")
                    break
                
                logger.info(f"Retrying with batch size {current_batch_size}...")
                config.error_handler.force_garbage_collection()
            else:
                logger.error("Maximum retries exceeded. Training failed.")
                break
    
    # Return dummy predictions if all retries failed
    logger.warning("Returning dummy predictions due to training failure")
    n_samples = len(test_df)
    n_targets = len(config.TARGET_COLS)
    return np.zeros((n_samples, n_targets))


def train_one_epoch_robust(model, loader, optimizer, device):
    """Robust training function with comprehensive error handling."""
    model.train()
    total_loss = 0
    total_samples = 0
    failed_batches = 0
    
    for batch_idx, data in enumerate(tqdm(loader, desc="Training", leave=False)):
        if data is None:
            failed_batches += 1
            continue
        
        try:
            data = config.device_manager.safe_device_transfer(data, device)
            optimizer.zero_grad()
            
            out = model(data)
            loss = masked_mse_loss(out, data.y, data.mask)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected in batch {batch_idx}: {loss}")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs
            
        except Exception as e:
            failed_batches += 1
            logger.warning(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    if failed_batches > 0:
        logger.warning(f"Failed to process {failed_batches} batches during training")
    
    if total_samples == 0:
        logger.error("No samples processed during training epoch")
        return float('inf')
    
    return total_loss / total_samples


@torch.no_grad()
def predict_robust(model, loader, device):
    """Generate predictions with robust error handling."""
    model.eval()
    all_ids = []
    all_preds = []
    failed_batches = 0
    
    for batch_idx, data in enumerate(tqdm(loader, desc="Predicting", leave=False)):
        if data is None:
            failed_batches += 1
            continue
        
        try:
            data = config.device_manager.safe_device_transfer(data, device)
            out = model(data)
            
            # Handle IDs
            if hasattr(data, 'id'):
                if torch.is_tensor(data.id):
                    all_ids.extend(data.id.tolist())
                elif isinstance(data.id, (list, tuple)):
                    all_ids.extend(data.id)
                else:
                    all_ids.append(data.id)
            else:
                # Fallback: use batch indices
                batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
                all_ids.extend(range(len(all_ids), len(all_ids) + batch_size))
            
            all_preds.append(out.cpu())
            
        except Exception as e:
            failed_batches += 1
            logger.warning(f"Error in prediction batch {batch_idx}: {str(e)}")
            continue
    
    if failed_batches > 0:
        logger.warning(f"Failed to process {failed_batches} batches during prediction")
    
    if not all_preds:
        logger.error("No predictions generated")
        # Return dummy predictions
        n_samples = len(all_ids) if all_ids else 1
        n_targets = len(config.TARGET_COLS)
        return list(range(n_samples)), np.zeros((n_samples, n_targets))
    
    predictions = torch.cat(all_preds, dim=0).numpy()
    return all_ids, predictions


def main_robust():
    """Main robust pipeline with comprehensive error handling and performance optimization."""
    logger.info("Starting robust polymer prediction pipeline with performance optimization...")
    
    try:
        # Load data with error handling
        train_df, test_df = load_data_robust()
        
        if train_df is None or test_df is None:
            raise ValueError("Failed to load data")
        
        # Create performance configuration
        perf_config = PerformanceConfig(
            enable_graph_cache=True,
            enable_memory_monitoring=True,
            enable_cpu_optimization=True,
            enable_progress_tracking=True,
            cache_dir="cache/robust_pipeline",
            checkpoint_dir="checkpoints/robust_pipeline",
            enable_performance_logging=True
        )
        
        logger.info("Using optimized pipeline for enhanced performance...")
        
        # Run optimized training pipeline
        results = run_optimized_training(
            train_df=train_df,
            test_df=test_df,
            target_cols=config.TARGET_COLS,
            num_epochs=config.NUM_EPOCHS,
            batch_size=None,  # Auto-optimized
            learning_rate=config.LEARNING_RATE,
            config=perf_config.__dict__,
            output_dir="outputs/robust_optimized"
        )
        
        # Extract predictions from results
        if results['predictions'] and 'predictions' in results['predictions']:
            gcn_preds = results['predictions']['predictions']
            test_ids = results['predictions']['ids']
            
            # Create submission with validation
            submission = pd.DataFrame({'id': test_ids})
            for i, col in enumerate(config.TARGET_COLS):
                if i < gcn_preds.shape[1]:
                    submission[col] = gcn_preds[:, i]
                else:
                    submission[col] = 0.0  # Fallback value
            
            # Validate submission format
            required_cols = ['id'] + config.TARGET_COLS
            submission = config.input_validator.validate_dataframe(
                submission, required_cols, "Submission DataFrame"
            )
            
            # Save submission
            output_path = 'submission_robust_optimized.csv'
            submission.to_csv(output_path, index=False)
            logger.info(f"Optimized robust submission saved to {output_path}")
        else:
            logger.warning("No predictions generated, falling back to basic robust training...")
            # Fallback to original robust training
            gcn_preds = train_gcn_robust(train_df, test_df)
            
            # Create submission with validation
            submission = pd.DataFrame({'id': test_df['id']})
            for i, col in enumerate(config.TARGET_COLS):
                submission[col] = gcn_preds[:, i]
            
            # Validate submission format
            required_cols = ['id'] + config.TARGET_COLS
            submission = config.input_validator.validate_dataframe(
                submission, required_cols, "Submission DataFrame"
            )
            
            # Save submission
            output_path = 'submission_robust_fallback.csv'
            submission.to_csv(output_path, index=False)
            logger.info(f"Fallback robust submission saved to {output_path}")
        
        # Print comprehensive results
        if 'performance_report' in results:
            perf_report = results['performance_report']
            logger.info(f"Performance Report:")
            logger.info(f"  System Info: {perf_report.get('system_info', {})}")
            logger.info(f"  Cache Stats: Hit ratio = {perf_report.get('cache_stats', {}).get('hit_ratio', 0):.2%}")
            
            if 'memory_stats' in perf_report and perf_report['memory_stats']:
                memory_stats = perf_report['memory_stats']
                logger.info(f"  Memory Stats: Cleanup count = {memory_stats.get('cleanup_count', 0)}")
        
        # Print training results
        if 'training_results' in results:
            training_results = results['training_results']
            logger.info(f"Training Results:")
            logger.info(f"  Success: {training_results.get('success', False)}")
            logger.info(f"  Epochs completed: {training_results.get('epochs_completed', 0)}")
            logger.info(f"  Best loss: {training_results.get('best_loss', 'N/A')}")
            logger.info(f"  Training duration: {training_results.get('training_duration', 0):.2f}s")
        
        # Print error summary
        error_summary = config.error_handler.get_error_summary()
        logger.info(f"Error summary: {error_summary}")
        
        # Print device info
        device_info = config.device_manager.get_device_info()
        logger.info(f"Device info: {device_info}")
        
        logger.info("Robust optimized pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Robust optimized pipeline failed: {str(e)}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        # Fallback to basic robust training
        logger.info("Attempting fallback to basic robust training...")
        try:
            train_df, test_df = load_data_robust()
            gcn_preds = train_gcn_robust(train_df, test_df)
            
            # Create submission
            submission = pd.DataFrame({'id': test_df['id']})
            for i, col in enumerate(config.TARGET_COLS):
                submission[col] = gcn_preds[:, i]
            
            output_path = 'submission_robust_emergency_fallback.csv'
            submission.to_csv(output_path, index=False)
            logger.info(f"Emergency fallback submission saved to {output_path}")
            
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
        
        # Print final error summary
        error_summary = config.error_handler.get_error_summary()
        logger.error(f"Final error summary: {error_summary}")
        
        raise


def test_robust_integration():
    """Test the robust integration with comprehensive error handling."""
    logger.info("Testing robust integration...")
    
    try:
        # Test error handling components
        logger.info("Testing error handling components...")
        
        # Test SMILES validation
        test_smiles = ['CCO', 'invalid_smiles', 'c1ccccc1', '', 'CC(C)O']
        validity_list, invalid_indices = config.smiles_validator.validate_smiles_list(test_smiles)
        logger.info(f"SMILES validation test: {sum(validity_list)}/{len(test_smiles)} valid")
        
        # Test memory management
        memory_info = config.memory_manager.error_handler.get_memory_info()
        logger.info(f"Memory info: {memory_info}")
        
        # Test device management
        device_info = config.device_manager.get_device_info()
        logger.info(f"Device info: {device_info}")
        
        # Create sample data for testing
        sample_data = {
            'id': [1, 2, 3, 4, 5],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O', 'C1=CC=CC=C1O', 'CCCCCCCC'],
            'Tg': [1.0, 2.0, None, 3.0, 4.0],
            'FFV': [0.1, 0.2, 0.3, None, 0.5],
            'Tc': [10.0, None, 30.0, 40.0, 50.0],
            'Density': [0.8, 0.9, 1.0, 1.1, None],
            'Rg': [2.0, 3.0, 4.0, 5.0, 6.0]
        }
        
        sample_df = pd.DataFrame(sample_data)
        
        # Test robust dataset creation
        dataset = RobustPolymerDataset(
            sample_df, 
            target_cols=config.TARGET_COLS, 
            is_test=False,
            error_handler=config.error_handler
        )
        logger.info(f"Robust dataset created: {len(dataset)} valid samples")
        
        # Test robust DataLoader
        loader = create_robust_dataloader(dataset, batch_size=2, error_handler=config.error_handler)
        
        # Test model with robust error handling
        sample_data = None
        for data in dataset:
            if data is not None:
                sample_data = data
                break
        
        if sample_data is not None:
            num_atom_features = sample_data.x.size(1)
            model = RobustPolymerGCN(num_atom_features=num_atom_features)
            model = config.device_manager.safe_device_transfer(model)
            model.eval()
            
            for batch in loader:
                if batch is not None:
                    with torch.no_grad():
                        batch = config.device_manager.safe_device_transfer(batch)
                        output = model(batch)
                    logger.info(f"Model output shape: {output.shape}")
                    break
            
            logger.info("Robust integration test completed successfully!")
            return True
        else:
            logger.error("No valid samples found in robust dataset")
            return False
            
    except Exception as e:
        logger.error(f"Robust integration test failed: {str(e)}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    # Test robust integration first
    if test_robust_integration():
        logger.info("Robust integration test passed! Running main pipeline...")
        main_robust()
    else:
        logger.error("Robust integration test failed! Please fix issues before running main pipeline.")