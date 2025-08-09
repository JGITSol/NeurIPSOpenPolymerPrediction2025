"""
Integrated performance optimization pipeline for polymer prediction.

This module provides a complete performance-optimized pipeline that integrates
all optimization components for maximum efficiency and resource utilization.
"""

import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from loguru import logger

from polymer_prediction.utils.performance import (
    PerformanceOptimizer,
    GraphCache,
    CPUOptimizer,
    MemoryMonitor,
    ProgressTracker
)
from polymer_prediction.data.optimized_dataloader import (
    CachedGraphDataset,
    OptimizedDataLoader,
    create_optimized_dataloader
)
from polymer_prediction.training.optimized_trainer import (
    OptimizedTrainer,
    create_optimized_trainer
)
from polymer_prediction.utils.error_handling import ErrorHandler, global_error_handler


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization features."""
    
    # Caching configuration
    enable_graph_cache: bool = True
    cache_dir: str = "cache"
    max_memory_cache_items: int = 10000
    max_disk_cache_gb: float = 5.0
    enable_disk_cache: bool = True
    
    # Memory optimization
    enable_memory_monitoring: bool = True
    memory_threshold: float = 0.85
    memory_check_interval: float = 30.0
    enable_adaptive_batching: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 128
    
    # CPU optimization
    enable_cpu_optimization: bool = True
    auto_detect_resources: bool = True
    force_cpu_mode: bool = False
    
    # Progress tracking
    enable_progress_tracking: bool = True
    progress_log_interval: int = 10
    enable_eta_estimation: bool = True
    
    # Training optimization
    enable_gradient_checkpointing: bool = False
    enable_mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 10
    save_best_only: bool = False
    
    # Logging and monitoring
    enable_performance_logging: bool = True
    log_level: str = "INFO"
    enable_resource_monitoring: bool = True


class OptimizedPipeline:
    """
    Complete performance-optimized pipeline for polymer prediction.
    
    This class integrates all performance optimization components to provide
    a seamless, high-performance training and inference pipeline.
    """
    
    def __init__(self, 
                 config: Optional[PerformanceConfig] = None,
                 error_handler: Optional[ErrorHandler] = None):
        """
        Initialize optimized pipeline.
        
        Args:
            config: Performance configuration
            error_handler: Error handler instance
        """
        self.config = config or PerformanceConfig()
        self.error_handler = error_handler or global_error_handler
        
        # Initialize performance components
        self._initialize_components()
        
        # Pipeline state
        self.is_initialized = False
        self.training_history = []
        self.performance_metrics = {}
        
        logger.info("OptimizedPipeline initialized with performance optimization features")
    
    def _initialize_components(self):
        """Initialize all performance optimization components."""
        
        # Performance optimizer (main coordinator)
        self.performance_optimizer = PerformanceOptimizer(
            cache_dir=self.config.cache_dir,
            enable_monitoring=self.config.enable_memory_monitoring,
            error_handler=self.error_handler
        )
        
        # Graph cache
        self.graph_cache = GraphCache(
            cache_dir=Path(self.config.cache_dir) / "graphs",
            max_memory_items=self.config.max_memory_cache_items,
            max_disk_size_gb=self.config.max_disk_cache_gb,
            enable_disk_cache=self.config.enable_disk_cache,
            error_handler=self.error_handler
        )
        
        # CPU optimizer
        self.cpu_optimizer = CPUOptimizer(error_handler=self.error_handler)
        
        # Memory monitor
        if self.config.enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor(
                memory_threshold=self.config.memory_threshold,
                check_interval=self.config.memory_check_interval,
                error_handler=self.error_handler
            )
            self.memory_monitor.start_monitoring()
        else:
            self.memory_monitor = None
        
        # Progress tracker
        if self.config.enable_progress_tracking:
            self.progress_tracker = ProgressTracker(
                name="OptimizedPipeline",
                log_interval=self.config.progress_log_interval,
                enable_eta=self.config.enable_eta_estimation
            )
        else:
            self.progress_tracker = None
        
        logger.info("Performance optimization components initialized")
    
    def prepare_data(self,
                    train_df: pd.DataFrame,
                    test_df: Optional[pd.DataFrame] = None,
                    target_cols: Optional[List[str]] = None,
                    batch_size: Optional[int] = None) -> Tuple[OptimizedDataLoader, Optional[OptimizedDataLoader]]:
        """
        Prepare optimized data loaders with caching and performance features.
        
        Args:
            train_df: Training DataFrame
            test_df: Optional test DataFrame
            target_cols: Target column names
            batch_size: Batch size (auto-optimized if None)
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        logger.info("Preparing optimized data loaders...")
        
        # Auto-optimize batch size if not provided
        if batch_size is None:
            batch_size = self.cpu_optimizer.get_optimal_batch_size()
        
        # Default target columns
        if target_cols is None:
            target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Create cached datasets
        train_dataset = CachedGraphDataset(
            df=train_df,
            target_cols=target_cols,
            is_test=False,
            error_handler=self.error_handler,
            graph_cache=self.graph_cache,
            enable_preprocessing_cache=self.config.enable_graph_cache,
            cache_dir=self.config.cache_dir
        )
        
        test_dataset = None
        if test_df is not None:
            test_dataset = CachedGraphDataset(
                df=test_df,
                target_cols=target_cols,
                is_test=True,
                error_handler=self.error_handler,
                graph_cache=self.graph_cache,
                enable_preprocessing_cache=self.config.enable_graph_cache,
                cache_dir=self.config.cache_dir
            )
        
        # Create optimized data loaders
        train_loader = OptimizedDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.cpu_optimizer.get_optimal_num_workers(),
            memory_monitor=self.memory_monitor,
            error_handler=self.error_handler,
            enable_adaptive_batching=self.config.enable_adaptive_batching,
            min_batch_size=self.config.min_batch_size,
            max_batch_size=self.config.max_batch_size
        )
        
        test_loader = None
        if test_dataset is not None:
            test_loader = OptimizedDataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.cpu_optimizer.get_optimal_num_workers(),
                memory_monitor=self.memory_monitor,
                error_handler=self.error_handler,
                enable_adaptive_batching=False  # No adaptive batching for test data
            )
        
        logger.info(f"Data loaders prepared: train_samples={len(train_dataset)}, "
                   f"test_samples={len(test_dataset) if test_dataset else 0}, "
                   f"batch_size={batch_size}")
        
        return train_loader, test_loader
    
    def create_optimized_model(self,
                              num_atom_features: int,
                              num_targets: int = 5,
                              model_config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Create model with CPU-optimized configuration.
        
        Args:
            num_atom_features: Number of atom features
            num_targets: Number of target properties
            model_config: Optional model configuration
            
        Returns:
            Optimized model
        """
        # Get CPU-optimized model configuration
        cpu_config = self.cpu_optimizer.get_cpu_model_config()
        
        if model_config:
            cpu_config.update(model_config)
        
        # Import model class (assuming it exists)
        try:
            from polymer_prediction.models.gcn import PolymerGCN
            
            # The existing PolymerGCN model has fixed parameters, so use them
            model = PolymerGCN(
                num_atom_features=num_atom_features,
                hidden_channels=cpu_config.get('hidden_channels', 128),
                num_gcn_layers=cpu_config.get('num_gcn_layers', 3)
            )
            
            logger.info(f"Created PolymerGCN model with {num_atom_features} atom features")
            
        except ImportError:
            # Fallback to a simple model if the specific model class doesn't exist
            logger.warning("PolymerGCN not found, using fallback model implementation")
            model = self._create_fallback_model(num_atom_features, num_targets, cpu_config)
        except Exception as e:
            # Handle any other errors in model creation
            logger.warning(f"Error creating PolymerGCN model: {e}. Using fallback model.")
            model = self._create_fallback_model(num_atom_features, num_targets, cpu_config)
        
        logger.info(f"Optimized model created with config: {cpu_config}")
        return model
    
    def _create_fallback_model(self, num_atom_features: int, num_targets: int, config: Dict[str, Any]) -> nn.Module:
        """Create a fallback model if the main model class is not available."""
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        class FallbackGCN(nn.Module):
            def __init__(self, num_atom_features, hidden_channels, num_gcn_layers, num_targets, dropout):
                super().__init__()
                
                self.convs = nn.ModuleList([GCNConv(num_atom_features, hidden_channels)])
                self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
                
                for _ in range(num_gcn_layers - 1):
                    self.convs.append(GCNConv(hidden_channels, hidden_channels))
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
                
                self.dropout = nn.Dropout(dropout)
                self.out = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_channels // 2, num_targets)
                )
                self.global_mean_pool = global_mean_pool
            
            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                
                for conv, bn in zip(self.convs, self.bns):
                    x = F.relu(bn(conv(x, edge_index)))
                    x = self.dropout(x)
                
                x = self.global_mean_pool(x, batch)
                return self.out(x)
        
        fallback_model = FallbackGCN(
            num_atom_features=num_atom_features,
            hidden_channels=config.get('hidden_channels', 128),
            num_gcn_layers=config.get('num_gcn_layers', 3),
            num_targets=num_targets,
            dropout=config.get('dropout', 0.2)
        )
        
        logger.info(f"Created fallback GCN model with {num_atom_features} atom features and {num_targets} targets")
        return fallback_model
    
    def train_model(self,
                   model: nn.Module,
                   train_loader: OptimizedDataLoader,
                   val_loader: Optional[OptimizedDataLoader] = None,
                   num_epochs: int = 50,
                   learning_rate: float = 1e-3,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Train model with comprehensive performance optimization.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device to use (auto-detected if None)
            
        Returns:
            Training results and performance metrics
        """
        logger.info(f"Starting optimized model training for {num_epochs} epochs")
        
        # Auto-detect device if not provided
        if device is None:
            device = self.performance_optimizer.device_manager.detect_optimal_device()
        
        # Create optimized trainer
        trainer = create_optimized_trainer(
            config=self.performance_optimizer.get_optimized_config(),
            enable_all_optimizations=True,
            cache_dir=self.config.cache_dir,
            checkpoint_dir=self.config.checkpoint_dir,
            error_handler=self.error_handler
        )
        
        # Create optimizer with CPU-optimized settings
        cpu_training_config = self.cpu_optimizer.get_cpu_training_config()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=cpu_training_config.get('weight_decay', 1e-4)
        )
        
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # Train with performance optimization
        with self.performance_optimizer.performance_context("model_training") as perf_context:
            results = trainer.train_model_optimized(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                num_epochs=num_epochs,
                val_loader=val_loader,
                scheduler=scheduler,
                save_checkpoints=self.config.enable_checkpointing,
                checkpoint_frequency=self.config.checkpoint_frequency
            )
        
        # Record training history
        self.training_history.append(results)
        
        # Update performance metrics
        self.performance_metrics.update({
            'last_training': results,
            'performance_report': self.performance_optimizer.get_performance_report(),
            'cache_stats': self.graph_cache.get_stats(),
            'memory_stats': self.memory_monitor.get_memory_stats() if self.memory_monitor else None
        })
        
        logger.info(f"Optimized training completed: best_loss={results.get('best_loss', 'N/A'):.4f}")
        
        return results
    
    def predict(self,
               model: nn.Module,
               test_loader: OptimizedDataLoader,
               device: Optional[torch.device] = None) -> Tuple[List[int], np.ndarray]:
        """
        Generate predictions with performance optimization.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to use
            
        Returns:
            Tuple of (ids, predictions)
        """
        logger.info("Generating optimized predictions...")
        
        if device is None:
            device = self.performance_optimizer.device_manager.detect_optimal_device()
        
        # Create optimized trainer for prediction
        trainer = OptimizedTrainer(
            performance_optimizer=self.performance_optimizer,
            error_handler=self.error_handler
        )
        
        # Generate predictions
        with self.performance_optimizer.performance_context("prediction") as perf_context:
            ids, predictions = trainer.predict_robust(model, test_loader, device)
        
        logger.info(f"Predictions generated: {len(ids)} samples, {predictions.shape[1]} targets")
        
        return ids, predictions
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'config': asdict(self.config),
            'system_info': {
                'cpu_count': self.cpu_optimizer.cpu_count,
                'memory_gb': self.cpu_optimizer.memory_gb,
                'cuda_available': torch.cuda.is_available()
            },
            'performance_optimizer': self.performance_optimizer.get_performance_report(),
            'cache_stats': self.graph_cache.get_stats(),
            'training_history_count': len(self.training_history),
            'current_metrics': self.performance_metrics
        }
        
        if self.memory_monitor:
            report['memory_stats'] = self.memory_monitor.get_memory_stats()
        
        return report
    
    def cleanup(self):
        """Clean up all resources."""
        logger.info("Cleaning up OptimizedPipeline resources...")
        
        # Stop memory monitoring
        if self.memory_monitor:
            self.memory_monitor.stop_monitoring()
        
        # Clear caches
        self.graph_cache.clear()
        
        # Cleanup performance optimizer
        if hasattr(self.performance_optimizer, 'cleanup'):
            self.performance_optimizer.cleanup()
        
        # Clear history to free memory
        self.training_history.clear()
        self.performance_metrics.clear()
        
        logger.info("OptimizedPipeline cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


def create_optimized_pipeline(config: Optional[Dict[str, Any]] = None,
                            enable_all_features: bool = True,
                            cache_dir: str = "cache",
                            checkpoint_dir: str = "checkpoints") -> OptimizedPipeline:
    """
    Create an optimized pipeline with all performance features.
    
    Args:
        config: Configuration dictionary
        enable_all_features: Whether to enable all optimization features
        cache_dir: Directory for caching
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        OptimizedPipeline instance
    """
    # Create performance configuration
    if config is None:
        perf_config = PerformanceConfig()
    else:
        perf_config = PerformanceConfig(**config)
    
    # Override directories
    perf_config.cache_dir = cache_dir
    perf_config.checkpoint_dir = checkpoint_dir
    
    # Disable features if requested
    if not enable_all_features:
        perf_config.enable_graph_cache = False
        perf_config.enable_memory_monitoring = False
        perf_config.enable_progress_tracking = False
        perf_config.enable_cpu_optimization = False
    
    # Create pipeline
    pipeline = OptimizedPipeline(config=perf_config)
    
    logger.info(f"Created optimized pipeline with all features {'enabled' if enable_all_features else 'disabled'}")
    
    return pipeline


def run_optimized_training(train_df: pd.DataFrame,
                          test_df: Optional[pd.DataFrame] = None,
                          target_cols: Optional[List[str]] = None,
                          num_epochs: int = 50,
                          batch_size: Optional[int] = None,
                          learning_rate: float = 1e-3,
                          config: Optional[Dict[str, Any]] = None,
                          output_dir: str = "outputs") -> Dict[str, Any]:
    """
    Run complete optimized training pipeline.
    
    Args:
        train_df: Training DataFrame
        test_df: Optional test DataFrame
        target_cols: Target column names
        num_epochs: Number of training epochs
        batch_size: Batch size (auto-optimized if None)
        learning_rate: Learning rate
        config: Optional configuration
        output_dir: Output directory
        
    Returns:
        Complete training results and performance metrics
    """
    logger.info("Starting complete optimized training pipeline...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create optimized pipeline
    with create_optimized_pipeline(
        config=config,
        cache_dir=str(output_path / "cache"),
        checkpoint_dir=str(output_path / "checkpoints")
    ) as pipeline:
        
        # Prepare data
        train_loader, test_loader = pipeline.prepare_data(
            train_df=train_df,
            test_df=test_df,
            target_cols=target_cols,
            batch_size=batch_size
        )
        
        # Get sample for model creation
        sample_data = None
        for data in train_loader.dataset:
            if data is not None:
                sample_data = data
                break
        
        if sample_data is None:
            raise ValueError("No valid samples found in training data")
        
        # Create optimized model
        num_atom_features = sample_data.x.size(1)
        num_targets = len(target_cols) if target_cols else 5
        
        model = pipeline.create_optimized_model(
            num_atom_features=num_atom_features,
            num_targets=num_targets
        )
        
        # Train model
        training_results = pipeline.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=None,  # Could split train_loader for validation
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        # Generate predictions if test data provided
        predictions = None
        if test_loader is not None:
            ids, pred_array = pipeline.predict(model, test_loader)
            predictions = {
                'ids': ids,
                'predictions': pred_array
            }
            
            # Save predictions
            if target_cols:
                pred_df = pd.DataFrame({'id': ids})
                for i, col in enumerate(target_cols):
                    pred_df[col] = pred_array[:, i]
                
                pred_path = output_path / "predictions.csv"
                pred_df.to_csv(pred_path, index=False)
                logger.info(f"Predictions saved to {pred_path}")
        
        # Get performance report
        performance_report = pipeline.get_performance_report()
        
        # Compile final results
        final_results = {
            'training_results': training_results,
            'predictions': predictions,
            'performance_report': performance_report,
            'config': config,
            'output_dir': str(output_path)
        }
        
        logger.info("Complete optimized training pipeline finished successfully!")
        
        return final_results