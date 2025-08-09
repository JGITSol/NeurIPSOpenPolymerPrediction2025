"""
Performance optimization package for polymer prediction.

This package provides comprehensive performance optimization features including:
- Graph caching system
- CPU-optimized configurations
- Memory monitoring and management
- Efficient data loading
- Progress tracking and logging
"""

from .performance_integration import (
    OptimizedPipeline,
    create_optimized_pipeline,
    PerformanceConfig,
    run_optimized_training
)

__all__ = [
    'OptimizedPipeline',
    'create_optimized_pipeline', 
    'PerformanceConfig',
    'run_optimized_training'
]