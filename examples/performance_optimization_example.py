"""
Comprehensive example demonstrating all performance optimization features.

This example shows how to use the complete performance optimization system
for efficient polymer prediction training and inference.
"""

import sys
import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.append('src')

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
from polymer_prediction.data.optimized_dataloader import (
    CachedGraphDataset,
    OptimizedDataLoader,
    create_optimized_dataloader
)
from polymer_prediction.training.optimized_trainer import create_optimized_trainer
from polymer_prediction.utils.logging import setup_logging


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample polymer data for demonstration."""
    logger.info(f"Creating sample data with {n_samples} samples...")
    
    # Sample SMILES strings (simplified for demonstration)
    sample_smiles = [
        'CCO',  # Ethanol
        'c1ccccc1',  # Benzene
        'CC(C)O',  # Isopropanol
        'C1=CC=CC=C1O',  # Phenol
        'CCCCCCCC',  # Octane
        'CC(=O)O',  # Acetic acid
        'CCN',  # Ethylamine
        'C1CCCCC1',  # Cyclohexane
        'CC(C)(C)O',  # tert-Butanol
        'C1=CC=C(C=C1)O'  # Phenol (alternative)
    ]
    
    # Generate sample data
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        smiles = np.random.choice(sample_smiles)
        
        # Generate synthetic target values with some correlation
        base_value = hash(smiles) % 100
        
        row = {
            'id': i,
            'SMILES': smiles,
            'Tg': base_value + np.random.normal(0, 10),  # Glass transition temperature
            'FFV': 0.1 + np.random.uniform(0, 0.4),      # Fractional free volume
            'Tc': base_value * 0.5 + np.random.normal(0, 5),  # Thermal conductivity
            'Density': 0.8 + np.random.uniform(0, 0.5),  # Density
            'Rg': 2.0 + np.random.uniform(0, 3.0)        # Radius of gyration
        }
        
        # Add some missing values randomly
        for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
            if np.random.random() < 0.1:  # 10% missing values
                row[col] = np.nan
        
        data.append(row)
    
    df = pd.DataFrame(data)
    logger.info(f"Sample data created: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def demonstrate_graph_caching():
    """Demonstrate graph caching system."""
    logger.info("=== Demonstrating Graph Caching System ===")
    
    # Create sample data
    sample_df = create_sample_data(100)
    
    # Initialize graph cache
    cache = GraphCache(
        cache_dir="cache/demo_graphs",
        max_memory_items=50,
        max_disk_size_gb=1.0,
        enable_disk_cache=True
    )
    
    # Create cached dataset
    dataset = CachedGraphDataset(
        df=sample_df,
        target_cols=['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
        is_test=False,
        graph_cache=cache,
        enable_preprocessing_cache=True
    )
    
    # Demonstrate cache performance
    logger.info("Testing cache performance...")
    
    # First pass - cache misses
    start_time = time.time()
    for i in range(min(20, len(dataset))):
        data = dataset[i]
    first_pass_time = time.time() - start_time
    
    # Second pass - cache hits
    start_time = time.time()
    for i in range(min(20, len(dataset))):
        data = dataset[i]
    second_pass_time = time.time() - start_time
    
    # Get cache statistics
    cache_stats = cache.get_stats()
    
    logger.info(f"Cache Performance Results:")
    logger.info(f"  First pass (cache misses): {first_pass_time:.3f}s")
    logger.info(f"  Second pass (cache hits): {second_pass_time:.3f}s")
    logger.info(f"  Speedup: {first_pass_time / max(second_pass_time, 0.001):.1f}x")
    logger.info(f"  Cache hit ratio: {cache_stats['hit_ratio']:.2%}")
    logger.info(f"  Memory items: {cache_stats['memory_items']}")
    logger.info(f"  Disk size: {cache_stats['disk_size_gb']:.3f}GB")
    
    # Cleanup
    cache.clear()


def demonstrate_cpu_optimization():
    """Demonstrate CPU optimization features."""
    logger.info("=== Demonstrating CPU Optimization ===")
    
    # Initialize CPU optimizer
    cpu_optimizer = CPUOptimizer()
    
    # Get system information
    logger.info(f"System Information:")
    logger.info(f"  CPU cores: {cpu_optimizer.cpu_count}")
    logger.info(f"  Memory: {cpu_optimizer.memory_gb:.1f}GB")
    
    # Get optimized configurations
    optimal_batch_size = cpu_optimizer.get_optimal_batch_size(32)
    optimal_workers = cpu_optimizer.get_optimal_num_workers()
    model_config = cpu_optimizer.get_cpu_model_config()
    training_config = cpu_optimizer.get_cpu_training_config()
    
    logger.info(f"CPU Optimizations:")
    logger.info(f"  Optimal batch size: {optimal_batch_size}")
    logger.info(f"  Optimal workers: {optimal_workers}")
    logger.info(f"  Model config: {model_config}")
    logger.info(f"  Training config: {training_config}")


def demonstrate_memory_monitoring():
    """Demonstrate memory monitoring system."""
    logger.info("=== Demonstrating Memory Monitoring ===")
    
    # Initialize memory monitor
    memory_monitor = MemoryMonitor(
        memory_threshold=0.8,
        check_interval=1.0  # Check every second for demo
    )
    
    # Start monitoring
    memory_monitor.start_monitoring()
    
    # Simulate memory usage
    logger.info("Simulating memory usage...")
    
    # Create some memory pressure
    large_arrays = []
    try:
        for i in range(5):
            # Create moderately sized arrays
            array = np.random.randn(1000, 1000)
            large_arrays.append(array)
            
            # Check memory stats
            memory_stats = memory_monitor.get_memory_stats()
            logger.info(f"  Step {i+1}: Memory usage = {memory_stats['current_memory']['system_usage']:.1%}")
            
            time.sleep(1)  # Allow monitoring to update
    
    finally:
        # Cleanup
        del large_arrays
        memory_monitor.stop_monitoring()
        
        # Get final stats
        final_stats = memory_monitor.get_memory_stats()
        logger.info(f"Memory Monitoring Results:")
        logger.info(f"  Cleanup count: {final_stats['cleanup_count']}")
        logger.info(f"  Alert count: {final_stats['alert_count']}")


def demonstrate_progress_tracking():
    """Demonstrate progress tracking system."""
    logger.info("=== Demonstrating Progress Tracking ===")
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(
        name="Demo Process",
        log_interval=5,
        enable_eta=True
    )
    
    # Simulate a long-running process
    total_steps = 50
    progress_tracker.start(total_steps)
    
    for step in range(total_steps):
        # Simulate work
        time.sleep(0.1)
        
        # Update progress with metrics
        metrics = {
            'loss': 1.0 - (step / total_steps) + np.random.normal(0, 0.1),
            'accuracy': step / total_steps + np.random.normal(0, 0.05)
        }
        
        progress_tracker.update(step + 1, metrics)
    
    # Finish tracking
    final_metrics = {
        'final_loss': 0.1,
        'final_accuracy': 0.95
    }
    progress_tracker.finish(final_metrics)
    
    # Get summary
    summary = progress_tracker.get_summary()
    logger.info(f"Progress Tracking Summary: {summary}")


def demonstrate_optimized_dataloader():
    """Demonstrate optimized data loader."""
    logger.info("=== Demonstrating Optimized DataLoader ===")
    
    # Create sample data
    sample_df = create_sample_data(200)
    
    # Create cached dataset first
    dataset = CachedGraphDataset(
        df=sample_df,
        target_cols=['Tg', 'FFV'],
        is_test=False,
        enable_preprocessing_cache=False  # Disable for demo speed
    )
    
    # Create optimized data loader
    dataloader = OptimizedDataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        enable_adaptive_batching=False  # Disable for demo stability
    )
    
    # Demonstrate data loading
    logger.info("Testing optimized data loading...")
    
    batch_count = 0
    total_samples = 0
    
    for batch in dataloader:
        if batch is not None:
            batch_count += 1
            total_samples += batch.num_graphs
            
            if batch_count >= 5:  # Process first 5 batches for demo
                break
    
    # Get performance statistics
    perf_stats = dataloader.get_performance_stats()
    
    logger.info(f"DataLoader Performance:")
    logger.info(f"  Batches processed: {batch_count}")
    logger.info(f"  Total samples: {total_samples}")
    logger.info(f"  Performance stats: {perf_stats}")


def demonstrate_complete_pipeline():
    """Demonstrate complete optimized pipeline."""
    logger.info("=== Demonstrating Complete Optimized Pipeline ===")
    
    # Create sample data
    train_df = create_sample_data(500)
    test_df = create_sample_data(100)
    
    # Create performance configuration
    config = PerformanceConfig(
        enable_graph_cache=True,
        enable_memory_monitoring=True,
        enable_cpu_optimization=True,
        enable_progress_tracking=True,
        cache_dir="cache/demo",
        checkpoint_dir="checkpoints/demo"
    )
    
    # Run complete optimized training
    results = run_optimized_training(
        train_df=train_df,
        test_df=test_df,
        target_cols=['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
        num_epochs=5,  # Short training for demo
        batch_size=None,  # Auto-optimized
        learning_rate=1e-3,
        config=config.__dict__,
        output_dir="outputs/demo"
    )
    
    logger.info(f"Complete Pipeline Results:")
    logger.info(f"  Training success: {results['training_results']['success']}")
    logger.info(f"  Epochs completed: {results['training_results']['epochs_completed']}")
    logger.info(f"  Best loss: {results['training_results']['best_loss']:.4f}")
    logger.info(f"  Predictions generated: {len(results['predictions']['ids']) if results['predictions'] else 0}")
    
    # Performance report
    perf_report = results['performance_report']
    logger.info(f"Performance Report:")
    logger.info(f"  System: {perf_report['system_info']}")
    logger.info(f"  Cache hit ratio: {perf_report['cache_stats']['hit_ratio']:.2%}")


def main():
    """Run all performance optimization demonstrations."""
    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/performance_demo.log")
    
    logger.info("Starting Performance Optimization Demonstration")
    logger.info("=" * 60)
    
    try:
        # Import time for demonstrations
        import time
        
        # Run all demonstrations
        demonstrate_graph_caching()
        logger.info("")
        
        demonstrate_cpu_optimization()
        logger.info("")
        
        demonstrate_memory_monitoring()
        logger.info("")
        
        demonstrate_progress_tracking()
        logger.info("")
        
        demonstrate_optimized_dataloader()
        logger.info("")
        
        demonstrate_complete_pipeline()
        logger.info("")
        
        logger.info("=" * 60)
        logger.info("Performance Optimization Demonstration Completed Successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    main()