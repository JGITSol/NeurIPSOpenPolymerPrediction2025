"""
Enhanced error handling and robustness utilities for polymer prediction.

This module provides comprehensive error handling for:
- Invalid SMILES strings with graceful continuation
- Memory management with automatic batch size reduction
- Model training failures with fallback mechanisms
- Device detection and automatic CPU/GPU fallback
- Input validation for all data processing steps
"""

import gc
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from rdkit import Chem
from loguru import logger
import warnings
import traceback
from functools import wraps
from contextlib import contextmanager


class PolymerPredictionError(Exception):
    """Base exception for polymer prediction errors."""
    pass


class SMILESValidationError(PolymerPredictionError):
    """Exception raised for invalid SMILES strings."""
    pass


class MemoryError(PolymerPredictionError):
    """Exception raised for memory-related issues."""
    pass


class ModelTrainingError(PolymerPredictionError):
    """Exception raised for model training failures."""
    pass


class DeviceError(PolymerPredictionError):
    """Exception raised for device-related issues."""
    pass


class InputValidationError(PolymerPredictionError):
    """Exception raised for input validation failures."""
    pass


class ErrorHandler:
    """Centralized error handling and robustness utilities."""
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize error handler.
        
        Args:
            log_level: Logging level for error reporting
        """
        self.log_level = log_level
        self.invalid_smiles_count = 0
        self.memory_warnings_count = 0
        self.device_fallbacks_count = 0
        self.training_failures_count = 0
        
    def handle_invalid_smiles(self, smiles: str, index: int, context: str = "") -> None:
        """Handle invalid SMILES strings with proper logging.
        
        Args:
            smiles: Invalid SMILES string
            index: Index of the invalid SMILES
            context: Additional context information
        """
        self.invalid_smiles_count += 1
        warning_msg = f"Invalid SMILES at index {index}: '{smiles}'"
        if context:
            warning_msg += f" (Context: {context})"
        
        logger.warning(warning_msg)
        
        # Log summary every 100 invalid SMILES
        if self.invalid_smiles_count % 100 == 0:
            logger.warning(f"Total invalid SMILES encountered so far: {self.invalid_smiles_count}")
    
    def handle_memory_error(self, current_batch_size: int, operation: str = "") -> int:
        """Handle memory errors with automatic batch size reduction.
        
        Args:
            current_batch_size: Current batch size
            operation: Description of the operation that failed
            
        Returns:
            New reduced batch size
        """
        self.memory_warnings_count += 1
        new_batch_size = max(1, current_batch_size // 2)
        
        warning_msg = f"Memory error during {operation}. Reducing batch size from {current_batch_size} to {new_batch_size}"
        logger.warning(warning_msg)
        
        # Force garbage collection
        self.force_garbage_collection()
        
        return new_batch_size
    
    def handle_training_failure(self, model_name: str, error: Exception, fallback_available: bool = False) -> bool:
        """Handle model training failures with fallback mechanisms.
        
        Args:
            model_name: Name of the model that failed
            error: The exception that occurred
            fallback_available: Whether a fallback mechanism is available
            
        Returns:
            True if should continue with fallback, False if should abort
        """
        self.training_failures_count += 1
        
        error_msg = f"Training failed for {model_name}: {str(error)}"
        logger.error(error_msg)
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        if fallback_available:
            logger.info(f"Attempting fallback mechanism for {model_name}")
            return True
        else:
            logger.error(f"No fallback available for {model_name}. Skipping this model.")
            return False
    
    def handle_device_error(self, preferred_device: torch.device, error: Exception) -> torch.device:
        """Handle device errors with automatic fallback.
        
        Args:
            preferred_device: The preferred device that failed
            error: The device-related error
            
        Returns:
            Fallback device to use
        """
        self.device_fallbacks_count += 1
        
        fallback_device = torch.device('cpu')
        
        warning_msg = f"Device error with {preferred_device}: {str(error)}. Falling back to {fallback_device}"
        logger.warning(warning_msg)
        
        return fallback_device
    
    def force_garbage_collection(self) -> None:
        """Force garbage collection and clear GPU cache if available."""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU cache")
        
        # Log memory usage
        memory_info = self.get_memory_info()
        logger.debug(f"Memory usage after cleanup: {memory_info}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information.
        
        Returns:
            Dictionary with memory usage statistics
        """
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_total_gb'] = system_memory.total / (1024**3)
        memory_info['system_available_gb'] = system_memory.available / (1024**3)
        memory_info['system_used_percent'] = system_memory.percent
        
        # GPU memory if available
        if torch.cuda.is_available():
            try:
                memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            except Exception as e:
                logger.debug(f"Could not get GPU memory info: {e}")
                memory_info['gpu_allocated_gb'] = 0.0
                memory_info['gpu_reserved_gb'] = 0.0
        
        return memory_info
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of all errors encountered.
        
        Returns:
            Dictionary with error counts
        """
        return {
            'invalid_smiles': self.invalid_smiles_count,
            'memory_warnings': self.memory_warnings_count,
            'device_fallbacks': self.device_fallbacks_count,
            'training_failures': self.training_failures_count
        }


class SMILESValidator:
    """Validator for SMILES strings with comprehensive error handling."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize SMILES validator.
        
        Args:
            error_handler: Error handler instance
        """
        self.error_handler = error_handler or ErrorHandler()
        self.validation_cache = {}
    
    def validate_smiles(self, smiles: str, index: Optional[int] = None) -> bool:
        """Validate a single SMILES string.
        
        Args:
            smiles: SMILES string to validate
            index: Optional index for error reporting
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(smiles, str):
            if index is not None:
                self.error_handler.handle_invalid_smiles(str(smiles), index, "Not a string")
            return False
        
        # Check cache first
        if smiles in self.validation_cache:
            return self.validation_cache[smiles]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            is_valid = mol is not None
            
            # Additional validation checks
            if is_valid:
                # Check for reasonable molecule size
                if mol.GetNumAtoms() == 0:
                    is_valid = False
                elif mol.GetNumAtoms() > 1000:  # Very large molecules might cause issues
                    logger.warning(f"Very large molecule with {mol.GetNumAtoms()} atoms: {smiles}")
            
            self.validation_cache[smiles] = is_valid
            
            if not is_valid and index is not None:
                self.error_handler.handle_invalid_smiles(smiles, index, "RDKit parsing failed")
            
            return is_valid
            
        except Exception as e:
            if index is not None:
                self.error_handler.handle_invalid_smiles(smiles, index, f"Exception: {str(e)}")
            self.validation_cache[smiles] = False
            return False
    
    def validate_smiles_list(self, smiles_list: List[str]) -> Tuple[List[bool], List[int]]:
        """Validate a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (validity_list, invalid_indices)
        """
        validity_list = []
        invalid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            is_valid = self.validate_smiles(smiles, i)
            validity_list.append(is_valid)
            if not is_valid:
                invalid_indices.append(i)
        
        logger.info(f"SMILES validation complete: {len(smiles_list) - len(invalid_indices)}/{len(smiles_list)} valid")
        
        return validity_list, invalid_indices
    
    def filter_valid_smiles(self, df: pd.DataFrame, smiles_column: str = 'SMILES') -> pd.DataFrame:
        """Filter DataFrame to keep only valid SMILES.
        
        Args:
            df: DataFrame containing SMILES
            smiles_column: Name of the SMILES column
            
        Returns:
            Filtered DataFrame with only valid SMILES
        """
        if smiles_column not in df.columns:
            raise InputValidationError(f"Column '{smiles_column}' not found in DataFrame")
        
        original_count = len(df)
        validity_list, invalid_indices = self.validate_smiles_list(df[smiles_column].tolist())
        
        # Filter out invalid SMILES
        valid_df = df[validity_list].reset_index(drop=True)
        
        logger.info(f"Filtered SMILES: kept {len(valid_df)}/{original_count} valid entries")
        
        return valid_df


class MemoryManager:
    """Memory management utilities with automatic optimization."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize memory manager.
        
        Args:
            error_handler: Error handler instance
        """
        self.error_handler = error_handler or ErrorHandler()
        self.initial_batch_size = None
        self.current_batch_size = None
        self.memory_threshold = 0.85  # 85% memory usage threshold
    
    def get_optimal_batch_size(self, initial_batch_size: int, available_memory_gb: float) -> int:
        """Calculate optimal batch size based on available memory.
        
        Args:
            initial_batch_size: Initial batch size to start with
            available_memory_gb: Available memory in GB
            
        Returns:
            Optimal batch size
        """
        # Rough estimation: reduce batch size if memory is limited
        if available_memory_gb < 2.0:  # Less than 2GB available
            optimal_batch_size = max(1, initial_batch_size // 4)
        elif available_memory_gb < 4.0:  # Less than 4GB available
            optimal_batch_size = max(1, initial_batch_size // 2)
        else:
            optimal_batch_size = initial_batch_size
        
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = optimal_batch_size
        
        logger.info(f"Optimal batch size: {optimal_batch_size} (available memory: {available_memory_gb:.1f}GB)")
        
        return optimal_batch_size
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits.
        
        Returns:
            True if memory usage is acceptable, False otherwise
        """
        memory_info = self.error_handler.get_memory_info()
        system_usage = memory_info.get('system_used_percent', 0) / 100.0
        
        if system_usage > self.memory_threshold:
            logger.warning(f"High memory usage detected: {system_usage:.1%}")
            return False
        
        return True
    
    def adaptive_batch_size_reduction(self, current_batch_size: int, operation: str = "") -> int:
        """Adaptively reduce batch size when memory issues occur.
        
        Args:
            current_batch_size: Current batch size
            operation: Description of the operation
            
        Returns:
            New reduced batch size
        """
        new_batch_size = self.error_handler.handle_memory_error(current_batch_size, operation)
        self.current_batch_size = new_batch_size
        return new_batch_size
    
    @contextmanager
    def memory_cleanup_context(self):
        """Context manager for automatic memory cleanup."""
        try:
            yield
        finally:
            self.error_handler.force_garbage_collection()


class DeviceManager:
    """Device management with automatic fallback capabilities."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize device manager.
        
        Args:
            error_handler: Error handler instance
        """
        self.error_handler = error_handler or ErrorHandler()
        self.preferred_device = None
        self.current_device = None
    
    def detect_optimal_device(self) -> torch.device:
        """Detect the optimal device for computation.
        
        Returns:
            Optimal device (GPU if available and working, otherwise CPU)
        """
        # Try GPU first
        if torch.cuda.is_available():
            try:
                # Test GPU functionality
                test_tensor = torch.randn(10, 10).cuda()
                _ = test_tensor @ test_tensor.T
                device = torch.device('cuda')
                logger.info(f"GPU detected and functional: {torch.cuda.get_device_name()}")
                
                # Check GPU memory
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU memory: {gpu_memory_gb:.1f}GB")
                
            except Exception as e:
                logger.warning(f"GPU available but not functional: {str(e)}")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for computation")
        
        self.preferred_device = device
        self.current_device = device
        return device
    
    def safe_device_transfer(self, tensor_or_model: Union[torch.Tensor, torch.nn.Module], 
                           target_device: Optional[torch.device] = None) -> Union[torch.Tensor, torch.nn.Module]:
        """Safely transfer tensor or model to device with error handling.
        
        Args:
            tensor_or_model: Tensor or model to transfer
            target_device: Target device (uses current device if None)
            
        Returns:
            Transferred tensor or model
        """
        if target_device is None:
            target_device = self.current_device
        
        try:
            return tensor_or_model.to(target_device)
        except Exception as e:
            # Fallback to CPU
            fallback_device = self.error_handler.handle_device_error(target_device, e)
            self.current_device = fallback_device
            return tensor_or_model.to(fallback_device)
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about current device setup.
        
        Returns:
            Dictionary with device information
        """
        info = {
            'preferred_device': str(self.preferred_device),
            'current_device': str(self.current_device),
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name()
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info


class InputValidator:
    """Comprehensive input validation for all data processing steps."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize input validator.
        
        Args:
            error_handler: Error handler instance
        """
        self.error_handler = error_handler or ErrorHandler()
    
    def validate_dataframe(self, df: pd.DataFrame, required_columns: List[str], 
                          name: str = "DataFrame") -> pd.DataFrame:
        """Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            name: Name of the DataFrame for error reporting
            
        Returns:
            Validated DataFrame
            
        Raises:
            InputValidationError: If validation fails
        """
        if not isinstance(df, pd.DataFrame):
            raise InputValidationError(f"{name} must be a pandas DataFrame")
        
        if df.empty:
            raise InputValidationError(f"{name} is empty")
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise InputValidationError(f"{name} missing required columns: {missing_columns}")
        
        logger.info(f"{name} validation passed: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def validate_file_path(self, file_path: Union[str, Path], must_exist: bool = True) -> Path:
        """Validate file path.
        
        Args:
            file_path: Path to validate
            must_exist: Whether the file must exist
            
        Returns:
            Validated Path object
            
        Raises:
            InputValidationError: If validation fails
        """
        path = Path(file_path)
        
        if must_exist and not path.exists():
            raise InputValidationError(f"File does not exist: {path}")
        
        if must_exist and not path.is_file():
            raise InputValidationError(f"Path is not a file: {path}")
        
        return path
    
    def validate_target_columns(self, df: pd.DataFrame, target_columns: List[str]) -> List[str]:
        """Validate target columns in DataFrame.
        
        Args:
            df: DataFrame containing target columns
            target_columns: List of target column names
            
        Returns:
            List of valid target columns
            
        Raises:
            InputValidationError: If no valid target columns found
        """
        valid_columns = []
        
        for col in target_columns:
            if col in df.columns:
                # Check if column has any non-null values
                if df[col].notna().any():
                    valid_columns.append(col)
                else:
                    logger.warning(f"Target column '{col}' has no non-null values")
            else:
                logger.warning(f"Target column '{col}' not found in DataFrame")
        
        if not valid_columns:
            raise InputValidationError("No valid target columns found")
        
        logger.info(f"Valid target columns: {valid_columns}")
        return valid_columns
    
    def validate_model_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model parameters.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Validated parameters dictionary
            
        Raises:
            InputValidationError: If validation fails
        """
        validated_params = {}
        
        # Common parameter validations
        if 'batch_size' in params:
            batch_size = params['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1:
                raise InputValidationError(f"batch_size must be a positive integer, got {batch_size}")
            validated_params['batch_size'] = batch_size
        
        if 'learning_rate' in params:
            lr = params['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise InputValidationError(f"learning_rate must be positive, got {lr}")
            validated_params['learning_rate'] = float(lr)
        
        if 'num_epochs' in params:
            epochs = params['num_epochs']
            if not isinstance(epochs, int) or epochs < 1:
                raise InputValidationError(f"num_epochs must be a positive integer, got {epochs}")
            validated_params['num_epochs'] = epochs
        
        # Copy other parameters as-is
        for key, value in params.items():
            if key not in validated_params:
                validated_params[key] = value
        
        return validated_params


def robust_function_wrapper(error_handler: Optional[ErrorHandler] = None, 
                          fallback_return: Any = None,
                          reraise: bool = False):
    """Decorator for adding robust error handling to functions.
    
    Args:
        error_handler: Error handler instance
        fallback_return: Value to return if function fails
        reraise: Whether to reraise the exception after handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(f"Full traceback: {traceback.format_exc()}")
                
                if reraise:
                    raise
                else:
                    return fallback_return
        
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()