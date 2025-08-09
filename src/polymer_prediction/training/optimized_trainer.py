"""
Optimized training module with comprehensive performance features.

This module provides enhanced training capabilities that integrate with
the performance optimization system for efficient model training.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
from loguru import logger
import gc
from contextlib import contextmanager

from polymer_prediction.training.robust_trainer import RobustTrainer
from polymer_prediction.training.trainer import masked_mse_loss
from polymer_prediction.utils.performance import (
    PerformanceOptimizer, 
    ProgressTracker, 
    MemoryMonitor,
    CPUOptimizer,
    PerformanceMetrics
)
from polymer_prediction.utils.error_handling import ErrorHandler, global_error_handler
from polymer_prediction.data.optimized_dataloader import OptimizedDataLoader


class OptimizedTrainer(RobustTrainer):
    """
    Enhanced trainer with comprehensive performance optimization features.
    
    Features:
    - Integrated performance monitoring and optimization
    - Automatic resource management and cleanup
    - Advanced progress tracking with ETA estimation
    - Memory-aware training with adaptive batch sizing
    - Comprehensive performance metrics collection
    - CPU-optimized training strategies
    """
    
    def __init__(self,
                 performance_optimizer: Optional[PerformanceOptimizer] = None,
                 enable_performance_tracking: bool = True,
                 enable_memory_optimization: bool = True,
                 enable_progress_tracking: bool = True,
                 checkpoint_dir: Optional[str] = "checkpoints",
                 **kwargs):
        """
        Initialize optimized trainer.
        
        Args:
            performance_optimizer: Performance optimizer instance
            enable_performance_tracking: Whether to enable performance tracking
            enable_memory_optimization: Whether to enable memory optimization
            enable_progress_tracking: Whether to enable detailed progress tracking
            checkpoint_dir: Directory for saving checkpoints
            **kwargs: Additional arguments for RobustTrainer
        """
        super().__init__(**kwargs)
        
        # Initialize performance components
        if performance_optimizer is None:
            self.performance_optimizer = PerformanceOptimizer(
                enable_monitoring=enable_memory_optimization,
                error_handler=self.error_handler
            )
        else:
            self.performance_optimizer = performance_optimizer
        
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_memory_optimization = enable_memory_optimization
        self.enable_progress_tracking = enable_progress_tracking
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.training_metrics = []
        self.epoch_metrics = []
        self.batch_metrics = []
        
        # Optimization state
        self.current_config = None
        self.performance_history = []
        
        logger.info("OptimizedTrainer initialized with performance optimization features")
    
    def train_model_optimized(self,
                             model: nn.Module,
                             train_loader: OptimizedDataLoader,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             num_epochs: int,
                             val_loader: Optional[OptimizedDataLoader] = None,
                             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                             save_checkpoints: bool = True,
                             checkpoint_frequency: int = 10) -> Dict[str, Any]:
        """
        Train model with comprehensive performance optimization.
        
        Args:
            model: Model to train
            train_loader: Optimized training data loader
            optimizer: Optimizer for training
            device: Device to use for training
            num_epochs: Number of training epochs
            val_loader: Optional validation data loader
            scheduler: Optional learning rate scheduler
            save_checkpoints: Whether to save model checkpoints
            checkpoint_frequency: Frequency of checkpoint saving
            
        Returns:
            Dictionary with training results and performance metrics
        """
        logger.info(f"Starting optimized model training for {num_epochs} epochs")
        
        # Get optimized configuration
        self.current_config = self.performance_optimizer.get_optimized_config()
        
        with self.performance_optimizer.performance_context("model_training") as perf_context:
            # Initialize progress tracking
            if self.enable_progress_tracking:
                progress_tracker = ProgressTracker("Training", log_interval=1)
                progress_tracker.start(num_epochs)
            
            # Training preparation
            training_start_time = time.time()
            model = self.device_manager.safe_device_transfer(model, device)
            
            # Training loop with performance optimization
            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                
                try:
                    # Memory cleanup before epoch
                    if self.enable_memory_optimization and epoch % 5 == 0:
                        self._perform_memory_cleanup(f"epoch_{epoch}_start")
                    
                    # Train one epoch with optimization
                    train_metrics = self._train_epoch_optimized(
                        model, train_loader, optimizer, device, epoch
                    )
                    
                    # Validation with optimization
                    val_metrics = None
                    if val_loader is not None:
                        val_metrics = self._validate_epoch_optimized(
                            model, val_loader, device, epoch
                        )
                    
                    # Learning rate scheduling
                    if scheduler is not None:
                        if val_metrics and 'loss' in val_metrics:
                            scheduler.step(val_metrics['loss'])
                        else:
                            scheduler.step(train_metrics['loss'])
                    
                    # Record epoch metrics
                    epoch_duration = time.time() - epoch_start_time
                    epoch_metrics = {
                        'epoch': epoch,
                        'duration': epoch_duration,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'memory_usage': self._get_current_memory_usage()
                    }
                    self.epoch_metrics.append(epoch_metrics)
                    
                    # Update progress tracking
                    if self.enable_progress_tracking:
                        progress_metrics = {
                            'train_loss': train_metrics['loss'],
                            'lr': optimizer.param_groups[0]['lr'],
                            'epoch_time': epoch_duration
                        }
                        if val_metrics:
                            progress_metrics['val_loss'] = val_metrics['loss']
                        
                        progress_tracker.update(epoch + 1, progress_metrics)
                    
                    # Early stopping check
                    current_loss = val_metrics['loss'] if val_metrics else train_metrics['loss']
                    if current_loss < self.best_loss:
                        self.best_loss = current_loss
                        self.patience_counter = 0
                        
                        # Save best model
                        if save_checkpoints and self.checkpoint_dir:
                            self._save_optimized_checkpoint(
                                model, optimizer, epoch, "best_model.pt"
                            )
                    else:
                        self.patience_counter += 1
                    
                    # Regular checkpoint saving
                    if (save_checkpoints and self.checkpoint_dir and 
                        epoch % checkpoint_frequency == 0):
                        self._save_optimized_checkpoint(
                            model, optimizer, epoch, f"checkpoint_epoch_{epoch}.pt"
                        )
                    
                    # Early stopping
                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
                    
                    # Performance analysis and optimization
                    if self.enable_performance_tracking and epoch % 10 == 0:
                        self._analyze_performance(epoch)
                
                except Exception as e:
                    logger.error(f"Error during epoch {epoch}: {e}")
                    if not self._handle_training_error(e, epoch):
                        break
            
            # Finalize training
            training_duration = time.time() - training_start_time
            
            if self.enable_progress_tracking:
                progress_tracker.finish({
                    'total_epochs': len(self.epoch_metrics),
                    'best_loss': self.best_loss,
                    'training_duration': training_duration
                })
            
            # Compile comprehensive results
            results = self._compile_training_results(training_duration, perf_context)
            
            logger.info(f"Optimized training completed: {len(self.epoch_metrics)} epochs, "
                       f"best_loss={self.best_loss:.4f}, duration={training_duration:.2f}s")
            
            return results
    
    def _train_epoch_optimized(self,
                              model: nn.Module,
                              loader: OptimizedDataLoader,
                              optimizer: torch.optim.Optimizer,
                              device: torch.device,
                              epoch: int) -> Dict[str, Any]:
        """Train one epoch with performance optimization."""
        model.train()
        
        # Initialize epoch tracking
        epoch_start_time = time.time()
        total_loss = 0.0
        total_samples = 0
        batch_times = []
        memory_usage = []
        
        # Batch processing with optimization
        for batch_idx, data in enumerate(loader):
            batch_start_time = time.time()
            
            if data is None:
                continue
            
            try:
                # Memory monitoring
                if self.enable_memory_optimization:
                    current_memory = self._get_current_memory_usage()
                    memory_usage.append(current_memory)
                    
                    # Adaptive memory management
                    if current_memory > 0.9:  # 90% memory usage
                        self._perform_memory_cleanup(f"epoch_{epoch}_batch_{batch_idx}")
                
                # Safe device transfer
                data = self.device_manager.safe_device_transfer(data, device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = masked_mse_loss(output, data.y, data.mask)
                
                # Validate loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss in epoch {epoch}, batch {batch_idx}: {loss}")
                    continue
                
                # Backward pass with gradient clipping
                loss.backward()
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                
                optimizer.step()
                
                # Record metrics
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                total_loss += loss.item() * data.num_graphs
                total_samples += data.num_graphs
                
                # Batch-level performance tracking
                if self.enable_performance_tracking:
                    batch_metrics = {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'loss': loss.item(),
                        'batch_time': batch_time,
                        'batch_size': data.num_graphs,
                        'memory_usage': current_memory if self.enable_memory_optimization else None
                    }
                    self.batch_metrics.append(batch_metrics)
                
            except Exception as e:
                logger.warning(f"Error in epoch {epoch}, batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch metrics
        epoch_duration = time.time() - epoch_start_time
        avg_loss = total_loss / max(1, total_samples)
        
        metrics = {
            'loss': avg_loss,
            'samples': total_samples,
            'duration': epoch_duration,
            'avg_batch_time': np.mean(batch_times) if batch_times else 0,
            'batches_processed': len(batch_times)
        }
        
        if memory_usage:
            metrics.update({
                'avg_memory_usage': np.mean(memory_usage),
                'max_memory_usage': np.max(memory_usage)
            })
        
        return metrics
    
    @torch.no_grad()
    def _validate_epoch_optimized(self,
                                 model: nn.Module,
                                 loader: OptimizedDataLoader,
                                 device: torch.device,
                                 epoch: int) -> Dict[str, Any]:
        """Validate one epoch with performance optimization."""
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        batch_times = []
        
        for batch_idx, data in enumerate(loader):
            if data is None:
                continue
            
            batch_start_time = time.time()
            
            try:
                data = self.device_manager.safe_device_transfer(data, device)
                output = model(data)
                loss = masked_mse_loss(output, data.y, data.mask)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item() * data.num_graphs
                    total_samples += data.num_graphs
                
                batch_times.append(time.time() - batch_start_time)
                
            except Exception as e:
                logger.warning(f"Error in validation epoch {epoch}, batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(1, total_samples)
        
        return {
            'loss': avg_loss,
            'samples': total_samples,
            'avg_batch_time': np.mean(batch_times) if batch_times else 0,
            'batches_processed': len(batch_times)
        }
    
    def _perform_memory_cleanup(self, context: str = ""):
        """Perform comprehensive memory cleanup."""
        if self.enable_memory_optimization:
            logger.debug(f"Performing memory cleanup: {context}")
            
            # Python garbage collection
            collected = gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force cleanup through performance optimizer
            self.performance_optimizer.memory_monitor.error_handler.force_garbage_collection()
            
            logger.debug(f"Memory cleanup completed: {collected} objects collected")
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except:
            return 0.0
    
    def _analyze_performance(self, epoch: int):
        """Analyze performance and suggest optimizations."""
        if not self.epoch_metrics:
            return
        
        recent_epochs = self.epoch_metrics[-5:]  # Last 5 epochs
        
        # Analyze training speed
        avg_epoch_time = np.mean([m['duration'] for m in recent_epochs])
        avg_batch_time = np.mean([m['train_metrics']['avg_batch_time'] for m in recent_epochs])
        
        # Analyze memory usage
        if self.enable_memory_optimization:
            memory_usage = [m['memory_usage'] for m in recent_epochs if m['memory_usage']]
            if memory_usage:
                avg_memory = np.mean(memory_usage)
                max_memory = np.max(memory_usage)
                
                if max_memory > 0.85:  # 85% memory usage
                    logger.warning(f"High memory usage detected: {max_memory:.1%}")
        
        # Log performance analysis
        logger.info(f"Performance analysis (epoch {epoch}): "
                   f"avg_epoch_time={avg_epoch_time:.2f}s, "
                   f"avg_batch_time={avg_batch_time:.4f}s")
    
    def _handle_training_error(self, error: Exception, epoch: int) -> bool:
        """Handle training errors with recovery mechanisms."""
        if "out of memory" in str(error).lower():
            logger.warning(f"Memory error in epoch {epoch}: {error}")
            
            # Perform aggressive memory cleanup
            self._perform_memory_cleanup(f"error_recovery_epoch_{epoch}")
            
            # Try to continue training
            return True
        else:
            logger.error(f"Unrecoverable error in epoch {epoch}: {error}")
            return False
    
    def _save_optimized_checkpoint(self,
                                  model: nn.Module,
                                  optimizer: torch.optim.Optimizer,
                                  epoch: int,
                                  filename: str):
        """Save checkpoint with performance metrics."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_path = self.checkpoint_dir / filename
        
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': self.best_loss,
                'training_history': self.training_history,
                'epoch_metrics': self.epoch_metrics,
                'performance_config': self.current_config,
                'performance_history': self.performance_history
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.debug(f"Optimized checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save optimized checkpoint: {e}")
    
    def _compile_training_results(self, 
                                 training_duration: float,
                                 perf_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive training results."""
        results = {
            'success': len(self.epoch_metrics) > 0,
            'epochs_completed': len(self.epoch_metrics),
            'best_loss': self.best_loss,
            'training_duration': training_duration,
            'performance_config': self.current_config
        }
        
        # Add epoch-level metrics
        if self.epoch_metrics:
            train_losses = [m['train_metrics']['loss'] for m in self.epoch_metrics]
            val_losses = [m['val_metrics']['loss'] for m in self.epoch_metrics 
                         if m['val_metrics'] is not None]
            
            results.update({
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1] if val_losses else None,
                'min_train_loss': min(train_losses),
                'min_val_loss': min(val_losses) if val_losses else None,
                'avg_epoch_duration': np.mean([m['duration'] for m in self.epoch_metrics])
            })
        
        # Add performance metrics
        if self.enable_performance_tracking:
            results['performance_metrics'] = {
                'total_batches_processed': len(self.batch_metrics),
                'avg_batch_time': np.mean([m['batch_time'] for m in self.batch_metrics]) if self.batch_metrics else 0,
                'performance_optimizer_stats': self.performance_optimizer.get_performance_report()
            }
        
        # Add data loader performance
        if hasattr(perf_context.get('train_loader'), 'get_performance_stats'):
            results['dataloader_performance'] = perf_context['train_loader'].get_performance_stats()
        
        return results
    
    @contextmanager
    def training_context(self, operation_name: str = "training"):
        """Context manager for training with automatic resource management."""
        logger.info(f"Starting training context: {operation_name}")
        
        try:
            # Initialize performance tracking
            if self.enable_performance_tracking:
                with self.performance_optimizer.performance_context(operation_name) as perf_context:
                    yield perf_context
            else:
                yield {}
        finally:
            # Cleanup resources
            self._perform_memory_cleanup("training_context_cleanup")
            logger.info(f"Training context completed: {operation_name}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary with performance metrics."""
        base_summary = super().get_training_summary()
        
        # Add performance-specific metrics
        performance_summary = {
            'performance_config': self.current_config,
            'epoch_count': len(self.epoch_metrics),
            'batch_count': len(self.batch_metrics),
            'performance_optimizer_report': self.performance_optimizer.get_performance_report()
        }
        
        # Merge summaries
        return {**base_summary, **performance_summary}
    
    def cleanup(self):
        """Clean up all resources."""
        logger.info("Cleaning up OptimizedTrainer resources...")
        
        # Clear metrics to free memory
        self.training_metrics.clear()
        self.epoch_metrics.clear()
        self.batch_metrics.clear()
        self.performance_history.clear()
        
        # Cleanup performance optimizer
        if hasattr(self.performance_optimizer, 'cleanup'):
            self.performance_optimizer.cleanup()
        
        # Final memory cleanup
        self._perform_memory_cleanup("trainer_cleanup")
        
        logger.info("OptimizedTrainer cleanup completed")


def create_optimized_trainer(config: Optional[Dict[str, Any]] = None,
                           enable_all_optimizations: bool = True,
                           cache_dir: str = "cache",
                           checkpoint_dir: str = "checkpoints",
                           error_handler: Optional[ErrorHandler] = None) -> OptimizedTrainer:
    """
    Create an optimized trainer with all performance features enabled.
    
    Args:
        config: Configuration dictionary
        enable_all_optimizations: Whether to enable all optimization features
        cache_dir: Directory for caching
        checkpoint_dir: Directory for checkpoints
        error_handler: Error handler instance
        
    Returns:
        OptimizedTrainer instance
    """
    error_handler = error_handler or global_error_handler
    
    # Create performance optimizer
    performance_optimizer = PerformanceOptimizer(
        cache_dir=cache_dir,
        enable_monitoring=enable_all_optimizations,
        error_handler=error_handler
    )
    
    # Get optimized configuration
    if config is None:
        config = performance_optimizer.get_optimized_config()
    else:
        config = performance_optimizer.get_optimized_config(config)
    
    # Create optimized trainer
    trainer = OptimizedTrainer(
        performance_optimizer=performance_optimizer,
        enable_performance_tracking=enable_all_optimizations,
        enable_memory_optimization=enable_all_optimizations,
        enable_progress_tracking=enable_all_optimizations,
        checkpoint_dir=checkpoint_dir,
        error_handler=error_handler,
        max_retries=config.get('training', {}).get('max_retries', 3),
        gradient_clip_norm=config.get('training', {}).get('gradient_clip_norm', 1.0)
    )
    
    logger.info(f"Created optimized trainer with all optimizations {'enabled' if enable_all_optimizations else 'disabled'}")
    
    return trainer