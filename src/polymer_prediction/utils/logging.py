"""Enhanced logging configuration for the polymer prediction project."""

import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

from loguru import logger


class StructuredLogger:
    """Structured logger with enhanced functionality for polymer prediction."""
    
    def __init__(self, name: str):
        """Initialize structured logger.
        
        Args:
            name: Logger name
        """
        self.name = name
        self.logger = logger.bind(name=name)
        self._context = {}
    
    def add_context(self, **kwargs):
        """Add context information to all log messages.
        
        Args:
            **kwargs: Context key-value pairs
        """
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear all context information."""
        self._context.clear()
    
    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Format message with context and extra information.
        
        Args:
            message: Log message
            extra: Additional information
            
        Returns:
            Formatted message
        """
        if not self._context and not extra:
            return message
        
        context_info = {}
        context_info.update(self._context)
        if extra:
            context_info.update(extra)
        
        if context_info:
            context_str = " | ".join([f"{k}={v}" for k, v in context_info.items()])
            return f"{message} | {context_str}"
        
        return message
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        formatted_msg = self._format_message(message, kwargs)
        self.logger.debug(formatted_msg)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        formatted_msg = self._format_message(message, kwargs)
        self.logger.info(formatted_msg)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        formatted_msg = self._format_message(message, kwargs)
        self.logger.warning(formatted_msg)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        formatted_msg = self._format_message(message, kwargs)
        self.logger.error(formatted_msg)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        formatted_msg = self._format_message(message, kwargs)
        self.logger.critical(formatted_msg)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance information.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional performance metrics
        """
        perf_info = {"operation": operation, "duration_seconds": f"{duration:.4f}"}
        perf_info.update(kwargs)
        
        message = f"Performance: {operation} completed in {duration:.4f}s"
        self.info(message, **perf_info)
    
    def log_memory_usage(self, stage: str, memory_info: Dict[str, Any]):
        """Log memory usage information.
        
        Args:
            stage: Current stage/operation
            memory_info: Memory usage information
        """
        message = f"Memory usage at {stage}"
        self.info(message, stage=stage, **memory_info)
    
    def log_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Log model performance metrics.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
        """
        message = f"Model metrics for {model_name}"
        self.info(message, model=model_name, **metrics)
    
    def log_data_statistics(self, stage: str, stats: Dict[str, Any]):
        """Log data statistics.
        
        Args:
            stage: Data processing stage
            stats: Statistics dictionary
        """
        message = f"Data statistics at {stage}"
        self.info(message, stage=stage, **stats)
    
    def log_configuration(self, config: Dict[str, Any]):
        """Log configuration information.
        
        Args:
            config: Configuration dictionary
        """
        message = "Configuration loaded"
        # Flatten nested config for logging
        flat_config = self._flatten_dict(config)
        self.info(message, **flat_config)
    
    def log_error_with_traceback(self, message: str, exception: Exception):
        """Log error with full traceback.
        
        Args:
            message: Error message
            exception: Exception object
        """
        import traceback
        
        error_info = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc()
        }
        
        self.error(message, **error_info)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for logging.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, logger: StructuredLogger, operation: str, **kwargs):
        """Initialize performance timer.
        
        Args:
            logger: Structured logger instance
            operation: Operation name
            **kwargs: Additional context
        """
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation}", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.log_performance(self.operation, duration, **self.context)
        else:
            self.logger.error(
                f"Operation {self.operation} failed after {duration:.4f}s",
                duration_seconds=f"{duration:.4f}",
                exception_type=exc_type.__name__ if exc_type else None,
                **self.context
            )


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    enable_structured_logging: bool = True,
    log_to_console: bool = True,
    log_to_file: bool = True,
) -> None:
    """Set up enhanced logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory to store log files
        enable_structured_logging: Enable structured logging features
        log_to_console: Enable console logging
        log_to_file: Enable file logging
    """
    # Remove default logger
    logger.remove()
    
    # Console handler
    if log_to_console:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[name]}</cyan> - "
            "<level>{message}</level>"
        )
        
        logger.add(
            sys.stderr,
            level=log_level,
            format=console_format,
            colorize=True,
            filter=lambda record: "name" in record["extra"]
        )
    
    # File handler
    if log_to_file and log_file:
        log_path = Path(log_dir) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if enable_structured_logging:
            # Structured text format (easier to handle than JSON)
            structured_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{extra[name]}:{function}:{line} | {message}"
            )
            
            logger.add(
                log_path,
                level=log_level,
                format=structured_format,
                rotation="10 MB",
                retention="1 week",
                compression="zip",
                filter=lambda record: "name" in record["extra"]
            )
        else:
            # Standard text format
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{extra[name]}:{function}:{line} - {message}"
            )
            
            logger.add(
                log_path,
                level=log_level,
                format=file_format,
                rotation="10 MB",
                retention="1 week",
                compression="zip",
                filter=lambda record: "name" in record["extra"]
            )
    
    logger.info(f"Enhanced logging initialized with level: {log_level}")


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


def get_performance_timer(logger: StructuredLogger, operation: str, **kwargs) -> PerformanceTimer:
    """Get a performance timer context manager.
    
    Args:
        logger: Structured logger instance
        operation: Operation name
        **kwargs: Additional context
        
    Returns:
        PerformanceTimer instance
    """
    return PerformanceTimer(logger, operation, **kwargs)


# Legacy compatibility
def setup_basic_logging(log_level: str = "INFO", log_file: Optional[str] = None, log_dir: str = "logs"):
    """Setup basic logging for backward compatibility."""
    setup_logging(
        log_level=log_level,
        log_file=log_file,
        log_dir=log_dir,
        enable_structured_logging=False
    )