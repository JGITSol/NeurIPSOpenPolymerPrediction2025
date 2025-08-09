"""
Tests for performance optimization features.

This module tests all performance optimization components including
caching, memory management, CPU optimization, and progress tracking.
"""

import pytest
import tempfile
import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.append('src')

from polymer_prediction.utils.performance import (
    GraphCache,
    CPUOptimizer,
    MemoryMonitor,
    ProgressTracker,
    PerformanceOptimizer,
    PerformanceMetrics
)
from polymer_prediction.data.optimized_dataloader import (
    CachedGraphDataset,
    OptimizedDataLoader,
    create_optimized_dataloader
)
from polymer_prediction.optimization import (
    create_optimized_pipeline,
    PerformanceConfig
)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    data = {
        'id': [1, 2, 3, 4, 5],
        'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O', 'C1=CC=CC=C1O', 'CCCCCCCC'],
        'Tg': [1.0, 2.0, None, 3.0, 4.0],
        'FFV': [0.1, 0.2, 0.3, None, 0.5],
        'Tc': [10.0, None, 30.0, 40.0, 50.0],
        'Density': [0.8, 0.9, 1.0, 1.1, None],
        'Rg': [2.0, 3.0, 4.0, 5.0, 6.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for caching tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestGraphCache:
    """Test graph caching system."""
    
    def test_cache_initialization(self, temp_cache_dir):
        """Test cache initialization."""
        cache = GraphCache(
            cache_dir=temp_cache_dir / "graphs",
            max_memory_items=100,
            max_disk_size_gb=1.0,
            enable_disk_cache=True
        )
        
        assert cache.cache_dir == temp_cache_dir / "graphs"
        assert cache.max_memory_items == 100
        assert cache.max_disk_size_gb == 1.0
        assert cache.enable_disk_cache is True
        assert len(cache.memory_cache) == 0
    
    def test_cache_operations(self, temp_cache_dir):
        """Test basic cache operations."""
        cache = GraphCache(
            cache_dir=temp_cache_dir / "graphs",
            max_memory_items=10,
            enable_disk_cache=False  # Memory only for this test
        )
        
        # Test cache miss
        result = cache.get("CCO")
        assert result is None
        
        # Test cache put and get
        test_data = {"x": torch.randn(5, 10), "edge_index": torch.randint(0, 5, (2, 8))}
        cache.put("CCO", test_data)
        
        retrieved = cache.get("CCO")
        assert retrieved is not None
        assert torch.equal(retrieved["x"], test_data["x"])
        assert torch.equal(retrieved["edge_index"], test_data["edge_index"])
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats["memory_hits"] == 1
        assert stats["misses"] == 1
        assert stats["memory_items"] == 1
    
    def test_cache_lru_eviction(self, temp_cache_dir):
        """Test LRU eviction policy."""
        cache = GraphCache(
            cache_dir=temp_cache_dir / "graphs",
            max_memory_items=2,  # Small cache for testing eviction
            enable_disk_cache=False
        )
        
        # Fill cache to capacity
        cache.put("smiles1", {"data": "test1"})
        cache.put("smiles2", {"data": "test2"})
        
        # Add one more item to trigger eviction
        cache.put("smiles3", {"data": "test3"})
        
        # First item should be evicted
        assert cache.get("smiles1") is None
        assert cache.get("smiles2") is not None
        assert cache.get("smiles3") is not None
        
        stats = cache.get_stats()
        assert stats["evictions"] >= 1
    
    def test_cache_clear(self, temp_cache_dir):
        """Test cache clearing."""
        cache = GraphCache(
            cache_dir=temp_cache_dir / "graphs",
            enable_disk_cache=False
        )
        
        # Add some data
        cache.put("test", {"data": "value"})
        assert len(cache.memory_cache) == 1
        
        # Clear cache
        cache.clear()
        assert len(cache.memory_cache) == 0
        assert cache.get("test") is None


class TestCPUOptimizer:
    """Test CPU optimization features."""
    
    def test_cpu_optimizer_initialization(self):
        """Test CPU optimizer initialization."""
        optimizer = CPUOptimizer()
        
        assert optimizer.cpu_count > 0
        assert optimizer.memory_gb > 0
    
    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation."""
        optimizer = CPUOptimizer()
        
        # Test with different base batch sizes
        optimal_16 = optimizer.get_optimal_batch_size(16)
        optimal_32 = optimizer.get_optimal_batch_size(32)
        optimal_64 = optimizer.get_optimal_batch_size(64)
        
        # Optimal batch sizes should be reasonable
        assert 1 <= optimal_16 <= 16
        assert 1 <= optimal_32 <= 32
        assert 1 <= optimal_64 <= 64
    
    def test_optimal_num_workers(self):
        """Test optimal number of workers calculation."""
        optimizer = CPUOptimizer()
        
        optimal_workers = optimizer.get_optimal_num_workers()
        
        # Should be between 0 and CPU count
        assert 0 <= optimal_workers <= optimizer.cpu_count
    
    def test_cpu_model_config(self):
        """Test CPU-optimized model configuration."""
        optimizer = CPUOptimizer()
        
        config = optimizer.get_cpu_model_config()
        
        # Check required keys
        required_keys = ['hidden_channels', 'num_gcn_layers', 'dropout', 'batch_norm']
        for key in required_keys:
            assert key in config
        
        # Check reasonable values
        assert config['hidden_channels'] > 0
        assert config['num_gcn_layers'] > 0
        assert 0 <= config['dropout'] <= 1
        assert isinstance(config['batch_norm'], bool)
    
    def test_cpu_training_config(self):
        """Test CPU-optimized training configuration."""
        optimizer = CPUOptimizer()
        
        config = optimizer.get_cpu_training_config()
        
        # Check required keys
        required_keys = ['num_epochs', 'learning_rate', 'weight_decay']
        for key in required_keys:
            assert key in config
        
        # Check reasonable values
        assert config['num_epochs'] > 0
        assert config['learning_rate'] > 0
        assert config['weight_decay'] >= 0


class TestMemoryMonitor:
    """Test memory monitoring system."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor(
            memory_threshold=0.8,
            check_interval=1.0
        )
        
        assert monitor.memory_threshold == 0.8
        assert monitor.check_interval == 1.0
        assert monitor.monitoring is False
    
    def test_memory_info_collection(self):
        """Test memory information collection."""
        monitor = MemoryMonitor()
        
        memory_info = monitor._get_memory_info()
        
        # Check required keys
        required_keys = ['timestamp', 'system_total_gb', 'system_available_gb', 
                        'system_usage', 'process_rss_mb', 'process_vms_mb']
        for key in required_keys:
            assert key in memory_info
        
        # Check reasonable values
        assert memory_info['system_total_gb'] > 0
        assert memory_info['system_available_gb'] > 0
        assert 0 <= memory_info['system_usage'] <= 1
        assert memory_info['process_rss_mb'] > 0
    
    def test_memory_monitoring_start_stop(self):
        """Test starting and stopping memory monitoring."""
        monitor = MemoryMonitor(check_interval=0.1)  # Fast interval for testing
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring is True
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.monitoring is False
    
    def test_memory_stats(self):
        """Test memory statistics collection."""
        monitor = MemoryMonitor()
        
        stats = monitor.get_memory_stats()
        
        # Check required keys
        required_keys = ['current_memory', 'monitoring_active', 'cleanup_count', 
                        'alert_count', 'memory_threshold']
        for key in required_keys:
            assert key in stats
        
        # Check reasonable values
        assert isinstance(stats['monitoring_active'], bool)
        assert stats['cleanup_count'] >= 0
        assert stats['alert_count'] >= 0


class TestProgressTracker:
    """Test progress tracking system."""
    
    def test_progress_tracker_initialization(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker(
            name="Test Process",
            log_interval=5,
            enable_eta=True
        )
        
        assert tracker.name == "Test Process"
        assert tracker.log_interval == 5
        assert tracker.enable_eta is True
        assert tracker.start_time is None
        assert tracker.current_step == 0
    
    def test_progress_tracking_workflow(self):
        """Test complete progress tracking workflow."""
        tracker = ProgressTracker("Test", log_interval=2)
        
        # Start tracking
        tracker.start(10)
        assert tracker.start_time is not None
        assert tracker.total_steps == 10
        
        # Update progress
        for i in range(5):
            tracker.update(i + 1, {"loss": 1.0 - i * 0.1})
            time.sleep(0.01)  # Small delay
        
        # Finish tracking
        tracker.finish({"final_loss": 0.5})
        
        # Check summary
        summary = tracker.get_summary()
        assert summary["current_step"] == 5
        assert summary["total_steps"] == 10
        assert "elapsed_time" in summary
        assert summary["metrics_count"] == 5
    
    def test_sub_tracker(self):
        """Test sub-tracker functionality."""
        main_tracker = ProgressTracker("Main Process")
        
        sub_tracker = main_tracker.add_sub_tracker("Sub Process", 5)
        
        assert "Sub Process" in main_tracker.sub_trackers
        assert sub_tracker.name == "Main Process/Sub Process"


class TestCachedGraphDataset:
    """Test cached graph dataset."""
    
    def test_cached_dataset_initialization(self, sample_dataframe, temp_cache_dir):
        """Test cached dataset initialization."""
        dataset = CachedGraphDataset(
            df=sample_dataframe,
            target_cols=['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
            is_test=False,
            cache_dir=temp_cache_dir / "graphs",
            enable_preprocessing_cache=False  # Skip preprocessing for speed
        )
        
        assert len(dataset.smiles_list) == 5
        assert dataset.target_cols == ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        assert dataset.is_test is False
    
    def test_cached_dataset_caching(self, sample_dataframe, temp_cache_dir):
        """Test dataset caching functionality."""
        dataset = CachedGraphDataset(
            df=sample_dataframe,
            target_cols=['Tg', 'FFV'],
            is_test=False,
            cache_dir=temp_cache_dir / "graphs",
            enable_preprocessing_cache=False
        )
        
        # Get cache statistics
        cache_stats = dataset.get_cache_stats()
        
        assert "dataset_cache_hits" in cache_stats
        assert "dataset_cache_misses" in cache_stats
        assert "dataset_total_requests" in cache_stats


class TestOptimizedDataLoader:
    """Test optimized data loader."""
    
    def test_optimized_dataloader_creation(self, sample_dataframe, temp_cache_dir):
        """Test optimized data loader creation."""
        dataset = CachedGraphDataset(
            df=sample_dataframe,
            target_cols=['Tg', 'FFV'],
            is_test=False,
            cache_dir=temp_cache_dir / "graphs",
            enable_preprocessing_cache=False
        )
        
        dataloader = OptimizedDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            enable_adaptive_batching=False  # Disable for testing
        )
        
        assert dataloader.current_batch_size == 2
        assert dataloader.shuffle is False
        assert dataloader.enable_adaptive_batching is False
    
    def test_dataloader_iteration(self, sample_dataframe, temp_cache_dir):
        """Test data loader iteration."""
        dataset = CachedGraphDataset(
            df=sample_dataframe,
            target_cols=['Tg', 'FFV'],
            is_test=False,
            cache_dir=temp_cache_dir / "graphs",
            enable_preprocessing_cache=False
        )
        
        dataloader = OptimizedDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            enable_adaptive_batching=False
        )
        
        batch_count = 0
        for batch in dataloader:
            if batch is not None:
                batch_count += 1
                assert hasattr(batch, 'x')
                assert hasattr(batch, 'edge_index')
                assert hasattr(batch, 'y')
                assert hasattr(batch, 'mask')
        
        assert batch_count > 0
    
    def test_dataloader_performance_stats(self, sample_dataframe, temp_cache_dir):
        """Test data loader performance statistics."""
        dataset = CachedGraphDataset(
            df=sample_dataframe,
            target_cols=['Tg'],
            is_test=False,
            cache_dir=temp_cache_dir / "graphs",
            enable_preprocessing_cache=False
        )
        
        dataloader = OptimizedDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            enable_adaptive_batching=False
        )
        
        # Process some batches
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Process first 2 batches
                break
        
        # Get performance stats
        stats = dataloader.get_performance_stats()
        
        required_keys = ['initial_batch_size', 'current_batch_size', 
                        'successful_batches', 'failed_batches']
        for key in required_keys:
            assert key in stats


class TestPerformanceOptimizer:
    """Test performance optimizer integration."""
    
    def test_performance_optimizer_initialization(self, temp_cache_dir):
        """Test performance optimizer initialization."""
        optimizer = PerformanceOptimizer(
            cache_dir=temp_cache_dir,
            enable_monitoring=False  # Disable for testing
        )
        
        assert optimizer.graph_cache is not None
        assert optimizer.cpu_optimizer is not None
        assert optimizer.memory_monitor is not None
    
    def test_optimized_config_generation(self, temp_cache_dir):
        """Test optimized configuration generation."""
        optimizer = PerformanceOptimizer(
            cache_dir=temp_cache_dir,
            enable_monitoring=False
        )
        
        config = optimizer.get_optimized_config()
        
        # Check required sections
        required_sections = ['model', 'training', 'data', 'caching', 'monitoring']
        for section in required_sections:
            assert section in config
        
        # Check model config
        assert 'hidden_channels' in config['model']
        assert 'num_gcn_layers' in config['model']
        
        # Check training config
        assert 'num_epochs' in config['training']
        assert 'learning_rate' in config['training']
        
        # Check data config
        assert 'batch_size' in config['data']
        assert 'num_workers' in config['data']
    
    def test_performance_context(self, temp_cache_dir):
        """Test performance context manager."""
        optimizer = PerformanceOptimizer(
            cache_dir=temp_cache_dir,
            enable_monitoring=False
        )
        
        with optimizer.performance_context("test_operation") as context:
            assert 'graph_cache' in context
            assert 'memory_monitor' in context
            assert 'cpu_optimizer' in context
            
            # Simulate some work
            time.sleep(0.1)
        
        # Check performance history
        assert len(optimizer.performance_history) > 0
        
        # Get performance report
        report = optimizer.get_performance_report()
        assert 'system_info' in report
        assert 'cache_stats' in report


class TestOptimizedPipeline:
    """Test complete optimized pipeline."""
    
    def test_pipeline_creation(self):
        """Test optimized pipeline creation."""
        config = PerformanceConfig(
            enable_memory_monitoring=False,  # Disable for testing
            enable_progress_tracking=False
        )
        
        pipeline = create_optimized_pipeline(
            config=config.__dict__,
            enable_all_features=False
        )
        
        assert pipeline is not None
        assert pipeline.config is not None
        assert pipeline.performance_optimizer is not None
    
    def test_pipeline_data_preparation(self, sample_dataframe):
        """Test pipeline data preparation."""
        config = PerformanceConfig(
            enable_memory_monitoring=False,
            enable_progress_tracking=False,
            enable_graph_cache=False  # Disable caching for speed
        )
        
        with create_optimized_pipeline(config=config.__dict__) as pipeline:
            train_loader, test_loader = pipeline.prepare_data(
                train_df=sample_dataframe,
                test_df=sample_dataframe.copy(),
                target_cols=['Tg', 'FFV'],
                batch_size=2
            )
            
            assert train_loader is not None
            assert test_loader is not None
            assert len(train_loader) > 0
            assert len(test_loader) > 0


class TestPerformanceMetrics:
    """Test performance metrics collection."""
    
    def test_performance_metrics_initialization(self):
        """Test performance metrics initialization."""
        metrics = PerformanceMetrics(start_time=time.time())
        
        assert metrics.start_time > 0
        assert metrics.end_time is None
        assert metrics.samples_processed == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
    
    def test_performance_metrics_finalization(self):
        """Test performance metrics finalization."""
        start_time = time.time()
        time.sleep(0.1)  # Small delay
        end_time = time.time()
        
        metrics = PerformanceMetrics(
            start_time=start_time,
            end_time=end_time,
            samples_processed=100,
            cache_hits=80,
            cache_misses=20
        )
        
        metrics.finalize()
        
        assert metrics.duration is not None
        assert metrics.duration > 0
        assert metrics.samples_per_second is not None
        assert metrics.samples_per_second > 0
        assert metrics.cache_hit_ratio == 0.8


# Integration tests
class TestPerformanceIntegration:
    """Test integration of all performance components."""
    
    def test_end_to_end_performance_optimization(self, sample_dataframe):
        """Test end-to-end performance optimization."""
        config = PerformanceConfig(
            enable_memory_monitoring=False,  # Disable for testing stability
            enable_progress_tracking=False,
            enable_graph_cache=False,  # Disable for speed
            enable_cpu_optimization=True
        )
        
        with create_optimized_pipeline(config=config.__dict__) as pipeline:
            # Prepare data
            train_loader, _ = pipeline.prepare_data(
                train_df=sample_dataframe,
                target_cols=['Tg', 'FFV'],
                batch_size=2
            )
            
            # Test data loading
            batch_count = 0
            for batch in train_loader:
                if batch is not None:
                    batch_count += 1
                    if batch_count >= 2:  # Process first 2 batches
                        break
            
            assert batch_count > 0
            
            # Get performance report
            report = pipeline.get_performance_report()
            assert 'system_info' in report
            assert 'cache_stats' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])