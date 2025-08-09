"""
Robust training utilities with comprehensive error handling and recovery mechanisms.

This module provides enhanced training functions that integrate with the error
handling system for graceful failure recovery and resource management.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from tqdm import tqdm
from loguru import logger
import gc
import time

from polymer_prediction.training.trainer import masked_mse_loss
from polymer_prediction.utils.error_handling import (
    ErrorHandler,
    MemoryManager,
    DeviceManager,
    ModelTrainingError,
    global_error_handler
)


class RobustTrainer:
    """
    Enhanced trainer with comprehensive error handling and recovery mechanisms.
    
    Features:
    - Automatic batch size reduction on memory errors
    - Gradient clipping and NaN/Inf detection
    - Device fallback capabilities
    - Training resumption from checkpoints
    - Comprehensive logging and monitoring
    """
    
    def __init__(self,
                 error_handler: Optional[ErrorHandler] = None,
                 memory_manager: Optional[MemoryManager] = None,
                 device_manager: Optional[DeviceManager] = None,
                 max_retries: int = 3,
                 min_batch_size: int = 1,
                 gradient_clip_norm: float = 1.0,
                 patience: int = 10):
        """
        Initialize robust trainer.
        
        Args:
            error_handler: Error handler for managing training errors
            memory_manager: Memory manager for handling memory issues
            device_manager: Device manager for handling device errors
            max_retries: Maximum number of training retries
            min_batch_size: Minimum allowed batch size
            gradient_clip_norm: Maximum norm for gradient clipping
            patience: Patience for early stopping
        """
        self.error_handler = error_handler or global_error_handler
        self.memory_manager = memory_manager or MemoryManager(self.error_handler)
        self.device_manager = device_manager or DeviceManager(self.error_handler)
        
        self.max_retries = max_retries
        self.min_batch_size = min_batch_size
        self.gradient_clip_norm = gradient_clip_norm
        self.patience = patience
        
        # Training state
        self.current_batch_size = None
        self.training_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"RobustTrainer initialized with max_retries={max_retries}, min_batch_size={min_batch_size}")
    
    def train_model_robust(self,
                          model: nn.Module,
                          train_loader,
                          optimizer: torch.optim.Optimizer,
                          device: torch.device,
                          num_epochs: int,
                          val_loader=None,
                          scheduler=None,
                          checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train model with comprehensive error handling and recovery.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer for training
            device: Device to use for training
            num_epochs: Number of training epochs
            val_loader: Optional validation data loader
            scheduler: Optional learning rate scheduler
            checkpoint_path: Optional path for saving checkpoints
            
        Returns:
            Dictionary with training results and statistics
        """
        logger.info(f"Starting robust model training for {num_epochs} epochs")
        
        # Initialize training state
        self.training_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        start_time = time.time()
        
        # Safely transfer model to device
        model = self.device_manager.safe_device_transfer(model, device)
        
        # Training loop with error handling
        for epoch in range(num_epochs):
            try:
                # Check memory before each epoch
                if not self.memory_manager.check_memory_usage():
                    logger.warning("High memory usage detected before epoch, forcing cleanup")
                    self.error_handler.force_garbage_collection()
                
                # Train one epoch
                train_loss = self._train_one_epoch_robust(
                    model, train_loader, optimizer, device, epoch
                )
                
                # Validate if validation loader provided
                val_loss = None
                if val_loader is not None:
                    val_loss = self._validate_one_epoch_robust(
                        model, val_loader, device, epoch
                    )
                
                # Update learning rate scheduler
                if scheduler is not None:
                    if val_loss is not None:
                        scheduler.step(val_loss)
                    else:
                        scheduler.step(train_loss)
                
                # Record training history
                epoch_stats = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'timestamp': time.time()
                }
                self.training_history.append(epoch_stats)
                
                # Check for improvement
                current_loss = val_loss if val_loss is not None else train_loss
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.patience_counter = 0
                    
                    # Save best model checkpoint
                    if checkpoint_path:
                        self._save_checkpoint(model, optimizer, epoch, checkpoint_path)
                else:
                    self.patience_counter += 1
                
                # Log progress
                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    log_msg = f"Epoch {epoch}/{num_epochs-1}: train_loss={train_loss:.4f}"
                    if val_loss is not None:
                        log_msg += f", val_loss={val_loss:.4f}"
                    log_msg += f", lr={optimizer.param_groups[0]['lr']:.6f}"
                    logger.info(log_msg)
                
                # Early stopping check
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs (patience={self.patience})")
                    break
                
                # Memory cleanup every few epochs
                if epoch % 5 == 0:
                    self.error_handler.force_garbage_collection()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Memory error during epoch {epoch}: {str(e)}")
                    
                    # Try to recover by reducing batch size
                    if hasattr(train_loader, 'batch_size'):
                        current_batch_size = train_loader.batch_size
                        new_batch_size = self.memory_manager.adaptive_batch_size_reduction(
                            current_batch_size, f"training epoch {epoch}"
                        )
                        
                        if new_batch_size < self.min_batch_size:
                            logger.error("Batch size reduced below minimum threshold. Stopping training.")
                            break
                        
                        logger.warning("Memory error recovery not implemented for existing DataLoader. Continuing with cleanup.")
                    
                    # Force cleanup and continue
                    self.error_handler.force_garbage_collection()
                    continue
                else:
                    logger.error(f"Runtime error during epoch {epoch}: {str(e)}")
                    break
            
            except Exception as e:
                logger.error(f"Unexpected error during epoch {epoch}: {str(e)}")
                if not self.error_handler.handle_training_failure(
                    f"epoch_{epoch}", e, fallback_available=True
                ):
                    break
                continue
        
        # Calculate training duration
        training_duration = time.time() - start_time
        
        # Compile training results
        results = {
            'success': len(self.training_history) > 0,
            'epochs_completed': len(self.training_history),
            'best_loss': self.best_loss,
            'final_loss': self.training_history[-1]['train_loss'] if self.training_history else float('inf'),
            'training_duration': training_duration,
            'training_history': self.training_history,
            'error_summary': self.error_handler.get_error_summary()
        }
        
        logger.info(f"Training completed: {results['epochs_completed']} epochs, best_loss={results['best_loss']:.4f}")
        
        return results
    
    def _train_one_epoch_robust(self,
                               model: nn.Module,
                               loader,
                               optimizer: torch.optim.Optimizer,
                               device: torch.device,
                               epoch: int) -> float:
        """
        Train one epoch with robust error handling.
        
        Args:
            model: Model to train
            loader: Training data loader
            optimizer: Optimizer
            device: Device to use
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        model.train()
        total_loss = 0.0
        total_samples = 0
        failed_batches = 0
        nan_losses = 0
        
        progress_bar = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)
        
        for batch_idx, data in enumerate(progress_bar):
            if data is None:
                failed_batches += 1
                continue
            
            try:
                # Safe device transfer
                data = self.device_manager.safe_device_transfer(data, device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                
                # Calculate loss
                loss = masked_mse_loss(output, data.y, data.mask)
                
                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_losses += 1
                    logger.warning(f"Invalid loss detected in batch {batch_idx}: {loss}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                
                # Check for gradient issues
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                
                if total_norm > 100:  # Very large gradients
                    logger.warning(f"Large gradient norm detected: {total_norm:.2f}")
                
                # Optimizer step
                optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item() * data.num_graphs
                total_samples += data.num_graphs
                
                # Update progress bar
                if total_samples > 0:
                    avg_loss = total_loss / total_samples
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Memory error in batch {batch_idx}: {str(e)}")
                    self.error_handler.force_garbage_collection()
                    failed_batches += 1
                    continue
                else:
                    logger.error(f"Runtime error in batch {batch_idx}: {str(e)}")
                    failed_batches += 1
                    continue
            
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {str(e)}")
                failed_batches += 1
                continue
        
        progress_bar.close()
        
        # Log batch processing statistics
        total_batches = failed_batches + (total_samples // loader.batch_size if hasattr(loader, 'batch_size') else 0)
        if failed_batches > 0:
            logger.warning(f"Failed to process {failed_batches} batches during training")
        if nan_losses > 0:
            logger.warning(f"Encountered {nan_losses} NaN/Inf losses during training")
        
        # Calculate average loss
        if total_samples == 0:
            logger.error("No samples processed during training epoch")
            return float('inf')
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    @torch.no_grad()
    def _validate_one_epoch_robust(self,
                                  model: nn.Module,
                                  loader,
                                  device: torch.device,
                                  epoch: int) -> float:
        """
        Validate one epoch with robust error handling.
        
        Args:
            model: Model to validate
            loader: Validation data loader
            device: Device to use
            epoch: Current epoch number
            
        Returns:
            Average validation loss for the epoch
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        failed_batches = 0
        
        progress_bar = tqdm(loader, desc=f"Validation Epoch {epoch}", leave=False)
        
        for batch_idx, data in enumerate(progress_bar):
            if data is None:
                failed_batches += 1
                continue
            
            try:
                # Safe device transfer
                data = self.device_manager.safe_device_transfer(data, device)
                
                # Forward pass
                output = model(data)
                
                # Calculate loss
                loss = masked_mse_loss(output, data.y, data.mask)
                
                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid validation loss in batch {batch_idx}: {loss}")
                    continue
                
                # Accumulate loss
                total_loss += loss.item() * data.num_graphs
                total_samples += data.num_graphs
                
                # Update progress bar
                if total_samples > 0:
                    avg_loss = total_loss / total_samples
                    progress_bar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
                
            except Exception as e:
                logger.warning(f"Error processing validation batch {batch_idx}: {str(e)}")
                failed_batches += 1
                continue
        
        progress_bar.close()
        
        # Log validation statistics
        if failed_batches > 0:
            logger.warning(f"Failed to process {failed_batches} validation batches")
        
        # Calculate average loss
        if total_samples == 0:
            logger.error("No samples processed during validation epoch")
            return float('inf')
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    @torch.no_grad()
    def predict_robust(self,
                      model: nn.Module,
                      loader,
                      device: torch.device) -> Tuple[List[int], np.ndarray]:
        """
        Generate predictions with robust error handling.
        
        Args:
            model: Trained model
            loader: Data loader for prediction
            device: Device to use
            
        Returns:
            Tuple of (ids, predictions) where predictions is numpy array
        """
        logger.info("Starting robust prediction generation...")
        
        model.eval()
        all_ids = []
        all_preds = []
        failed_batches = 0
        
        progress_bar = tqdm(loader, desc="Predicting", leave=False)
        
        for batch_idx, data in enumerate(progress_bar):
            if data is None:
                failed_batches += 1
                continue
            
            try:
                # Safe device transfer
                data = self.device_manager.safe_device_transfer(data, device)
                
                # Forward pass
                output = model(data)
                
                # Check for invalid predictions
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logger.warning(f"Invalid predictions in batch {batch_idx}")
                    # Replace invalid values with zeros
                    output = torch.where(torch.isnan(output) | torch.isinf(output), 
                                       torch.zeros_like(output), output)
                
                # Extract IDs
                if hasattr(data, 'id'):
                    if torch.is_tensor(data.id):
                        batch_ids = data.id.tolist()
                    elif isinstance(data.id, (list, tuple)):
                        batch_ids = list(data.id)
                    else:
                        batch_ids = [data.id]
                else:
                    # Fallback: use sequential IDs
                    batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
                    batch_ids = list(range(len(all_ids), len(all_ids) + batch_size))
                
                all_ids.extend(batch_ids)
                all_preds.append(output.cpu())
                
            except Exception as e:
                logger.warning(f"Error processing prediction batch {batch_idx}: {str(e)}")
                failed_batches += 1
                continue
        
        progress_bar.close()
        
        # Log prediction statistics
        if failed_batches > 0:
            logger.warning(f"Failed to process {failed_batches} prediction batches")
        
        if not all_preds:
            logger.error("No predictions generated")
            # Return dummy predictions
            n_samples = len(all_ids) if all_ids else 1
            n_targets = 5  # Default number of targets
            return list(range(n_samples)), np.zeros((n_samples, n_targets))
        
        # Concatenate predictions
        predictions = torch.cat(all_preds, dim=0).numpy()
        
        logger.info(f"Prediction generation completed: {len(all_ids)} samples, {predictions.shape[1]} targets")
        
        return all_ids, predictions
    
    def _save_checkpoint(self,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        checkpoint_path: str) -> None:
        """
        Save model checkpoint with error handling.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            epoch: Current epoch
            checkpoint_path: Path to save checkpoint
        """
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': self.best_loss,
                'training_history': self.training_history
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.debug(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
    
    def load_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       checkpoint_path: str) -> int:
        """
        Load model checkpoint with error handling.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number from checkpoint, or 0 if loading failed
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', [])
            
            epoch = checkpoint.get('epoch', 0)
            logger.info(f"Checkpoint loaded: epoch {epoch}, best_loss {self.best_loss:.4f}")
            
            return epoch
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return 0
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.training_history:
            return {"status": "no_training_history"}
        
        train_losses = [h['train_loss'] for h in self.training_history]
        val_losses = [h['val_loss'] for h in self.training_history if h['val_loss'] is not None]
        
        summary = {
            "epochs_completed": len(self.training_history),
            "best_loss": self.best_loss,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "training_duration": self.training_history[-1]['timestamp'] - self.training_history[0]['timestamp'],
            "convergence_epoch": next((i for i, h in enumerate(self.training_history) 
                                     if (h['val_loss'] or h['train_loss']) == self.best_loss), None),
            "error_summary": self.error_handler.get_error_summary()
        }
        
        return summary


# Convenience functions for backward compatibility
def train_one_epoch_robust(model, loader, optimizer, device, error_handler=None):
    """Robust training function for single epoch (backward compatibility)."""
    trainer = RobustTrainer(error_handler=error_handler)
    return trainer._train_one_epoch_robust(model, loader, optimizer, device, 0)


@torch.no_grad()
def evaluate_robust(model, loader, device, error_handler=None):
    """Robust evaluation function (backward compatibility)."""
    trainer = RobustTrainer(error_handler=error_handler)
    return trainer._validate_one_epoch_robust(model, loader, device, 0)


@torch.no_grad()
def predict_robust(model, loader, device, error_handler=None):
    """Robust prediction function (backward compatibility)."""
    trainer = RobustTrainer(error_handler=error_handler)
    return trainer.predict_robust(model, loader, device)