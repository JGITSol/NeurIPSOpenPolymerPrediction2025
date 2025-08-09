"""
Performance optimization utilities for polymer prediction.

This module provides comprehensive performance optimization features including:
- Graph caching system to avoid recomputing molecular graphs
- CPU-optimized batch sizes and model configurations
- Memory monitoring and automatic resource management
- Efficient data loading with preprocessing and caching mechanisms
- Progress tracking and logging for long-running training processes
"""

import os
import pickle
import hashlib
import time
import psutil
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from loguru import logger
import gc
import threading
from collections import defaultdict, deque

from polymer_prediction.utils.error_handling import ErrorHandler, global_error_handler


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    samples_processed: int = 0
    samples_per_second: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_ratio: Optional[float] = None
    
    def finalize(self):
        """Finalize metrics calculation."""
        if self.end_time is not None:
            self.duration = self.end_time - self.start_time
            if self.duration > 0 and self.samples_processed > 0:
                self.samples_per_second = self.samples_processed / self.duration
        
        if self.cache_hits + self.cache_misses > 0:
            self.cache_hit_ratio = self.cache_hits / (self.cache_hits + self.cache_misses)


class GraphCache:
    """
    High-performance graph caching system with disk persistence and memory management.
    
    Features:
    - LRU eviction policy for memory management
    - Disk persistence for large datasets
    - Hash-based cache keys for SMILES strings
    - Automatic cache size management
    - Thread-safe operations
    """
    
    def __init__(self,
                 cache_dir: Union[str, Path] = "cache/graphs",
                 max_memory_items: int = 10000,
                 max_disk_size_gb: float = 5.0,
                 enable_disk_cache: bool = True,
                 error_handler: Optional[ErrorHandler] = None):
        """
        Initialize graph cache.
        
        Args:
            cache_dir: Directory for disk cache storage
            max_memory_items: Maximum number of items in memory cache
            max_disk_size_gb: Maximum disk cache size in GB
            enable_disk_cache: Whether to enable disk caching
            error_handler: Error handler instance
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_items = max_memory_items
        self.max_disk_size_gb = max_disk_size_gb
        self.enable_disk_cache = enable_disk_cache
        self.error_handler = error_handler or global_error_handler
        
        # Create cache directory
        if self.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache with LRU eviction
        self.memory_cache = {}
        self.access_order = deque()  # For LRU tracking
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_writes': 0,
            'disk_reads': 0,
            'errors': 0
        }
        
        logger.info(f"GraphCache initialized: memory_limit={max_memory_items}, "
                   f"disk_cache={'enabled' if enable_disk_cache else 'disabled'}")
    
    def _get_cache_key(self, smiles: str) -> str:
        """Generate cache key for SMILES string."""
        return hashlib.md5(smiles.encode()).hexdigest()
    
    def _get_disk_path(self, cache_key: str) -> Path:
        """Get disk path for cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _evict_lru_item(self):
        """Evict least recently used item from memory cache."""
        if not self.access_order:
            return
        
        lru_key = self.access_order.popleft()
        if lru_key in self.memory_cache:
            del self.memory_cache[lru_key]
            self.stats['evictions'] += 1
    
    def _update_access_order(self, cache_key: str):
        """Update access order for LRU tracking."""
        # Remove from current position if exists
        try:
            self.access_order.remove(cache_key)
        except ValueError:
            pass
        
        # Add to end (most recently used)
        self.access_order.append(cache_key)
    
    def get(self, smiles: str) -> Optional[Any]:
        """
        Get cached graph for SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Cached graph data or None if not found
        """
        cache_key = self._get_cache_key(smiles)
        
        with self.cache_lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self._update_access_order(cache_key)
                self.stats['memory_hits'] += 1
                return self.memory_cache[cache_key]
            
            # Check disk cache if enabled
            if self.enable_disk_cache:
                disk_path = self._get_disk_path(cache_key)
                if disk_path.exists():
                    try:
                        with open(disk_path, 'rb') as f:
                            graph_data = pickle.load(f)
                        
                        # Add to memory cache
                        self._add_to_memory_cache(cache_key, graph_data)
                        self.stats['disk_hits'] += 1
                        self.stats['disk_reads'] += 1
                        return graph_data
                        
                    except Exception as e:
                        logger.warning(f"Error reading from disk cache: {e}")
                        self.stats['errors'] += 1
                        # Remove corrupted file
                        try:
                            disk_path.unlink()
                        except:
                            pass
            
            # Cache miss
            self.stats['misses'] += 1
            return None
    
    def put(self, smiles: str, graph_data: Any):
        """
        Store graph data in cache.
        
        Args:
            smiles: SMILES string
            graph_data: Graph data to cache
        """
        cache_key = self._get_cache_key(smiles)
        
        with self.cache_lock:
            # Add to memory cache
            self._add_to_memory_cache(cache_key, graph_data)
            
            # Add to disk cache if enabled
            if self.enable_disk_cache:
                self._add_to_disk_cache(cache_key, graph_data)
    
    def _add_to_memory_cache(self, cache_key: str, graph_data: Any):
        """Add item to memory cache with LRU eviction."""
        # Evict items if at capacity
        while len(self.memory_cache) >= self.max_memory_items:
            self._evict_lru_item()
        
        self.memory_cache[cache_key] = graph_data
        self._update_access_order(cache_key)
    
    def _add_to_disk_cache(self, cache_key: str, graph_data: Any):
        """Add item to disk cache."""
        try:
            # Check disk space
            if self._get_cache_size_gb() >= self.max_disk_size_gb:
                self._cleanup_disk_cache()
            
            disk_path = self._get_disk_path(cache_key)
            with open(disk_path, 'wb') as f:
                pickle.dump(graph_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.stats['disk_writes'] += 1
            
        except Exception as e:
            logger.warning(f"Error writing to disk cache: {e}")
            self.stats['errors'] += 1
    
    def _get_cache_size_gb(self) -> float:
        """Get current disk cache size in GB."""
        if not self.cache_dir.exists():
            return 0.0
        
        total_size = 0
        try:
            for file_path in self.cache_dir.glob("*.pkl"):
                total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating cache size: {e}")
        
        return total_size / (1024**3)
    
    def _cleanup_disk_cache(self):
        """Clean up disk cache by removing oldest files."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            if not cache_files:
                return
            
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda p: p.stat().st_mtime)
            
            # Remove oldest 20% of files
            files_to_remove = cache_files[:len(cache_files) // 5]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"Error removing cache file {file_path}: {e}")
            
            logger.info(f"Cleaned up {len(files_to_remove)} cache files")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def clear(self):
        """Clear all cached data."""
        with self.cache_lock:
            self.memory_cache.clear()
            self.access_order.clear()
            
            if self.enable_disk_cache and self.cache_dir.exists():
                try:
                    for file_path in self.cache_dir.glob("*.pkl"):
                        file_path.unlink()
                    logger.info("Disk cache cleared")
                except Exception as e:
                    logger.error(f"Error clearing disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            total_requests = sum([self.stats['memory_hits'], self.stats['disk_hits'], self.stats['misses']])
            
            stats = dict(self.stats)
            stats.update({
                'memory_items': len(self.memory_cache),
                'memory_capacity': self.max_memory_items,
                'disk_size_gb': self._get_cache_size_gb(),
                'disk_capacity_gb': self.max_disk_size_gb,
                'total_requests': total_requests,
                'hit_ratio': (self.stats['memory_hits'] + self.stats['disk_hits']) / max(1, total_requests),
                'memory_hit_ratio': self.stats['memory_hits'] / max(1, total_requests),
                'disk_hit_ratio': self.stats['disk_hits'] / max(1, total_requests)
            })
            
            return stats


class CPUOptimizer:
    """
    CPU-specific optimizations for efficient training on limited hardware.
    
    Features:
    - Automatic batch size optimization based on CPU cores and memory
    - CPU-friendly model configurations
    - Thread pool management for data loading
    - Memory-efficient training strategies
    """
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize CPU optimizer."""
        self.error_handler = error_handler or global_error_handler
        self.cpu_count = os.cpu_count() or 1
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        logger.info(f"CPU Optimizer initialized: {self.cpu_count} cores, {self.memory_gb:.1f}GB RAM")
    
    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """
        Calculate optimal batch size for CPU training.
        
        Args:
            base_batch_size: Base batch size to optimize from
            
        Returns:
            Optimized batch size
        """
        # Reduce batch size based on available memory
        if self.memory_gb < 4:
            optimal_batch_size = max(1, base_batch_size // 4)
        elif self.memory_gb < 8:
            optimal_batch_size = max(1, base_batch_size // 2)
        elif self.memory_gb < 16:
            optimal_batch_size = min(base_batch_size, 16)
        else:
            optimal_batch_size = min(base_batch_size, 32)
        
        # Further adjust based on CPU cores
        if self.cpu_count <= 2:
            optimal_batch_size = min(optimal_batch_size, 8)
        elif self.cpu_count <= 4:
            optimal_batch_size = min(optimal_batch_size, 16)
        
        logger.info(f"Optimal CPU batch size: {optimal_batch_size} (base: {base_batch_size})")
        return optimal_batch_size
    
    def get_optimal_num_workers(self, max_workers: int = 4) -> int:
        """
        Get optimal number of data loading workers.
        
        Args:
            max_workers: Maximum number of workers
            
        Returns:
            Optimal number of workers
        """
        # Use fewer workers on systems with limited cores
        if self.cpu_count <= 2:
            optimal_workers = 0  # Single-threaded for very limited systems
        elif self.cpu_count <= 4:
            optimal_workers = min(2, max_workers)
        else:
            optimal_workers = min(self.cpu_count // 2, max_workers)
        
        logger.info(f"Optimal data loading workers: {optimal_workers}")
        return optimal_workers
    
    def get_cpu_model_config(self) -> Dict[str, Any]:
        """
        Get CPU-optimized model configuration.
        
        Returns:
            Dictionary with CPU-optimized model parameters
        """
        config = {
            'hidden_channels': 64 if self.memory_gb < 8 else 128,
            'num_gcn_layers': 2 if self.memory_gb < 4 else 3,
            'dropout': 0.2,  # Slightly higher dropout for regularization
            'batch_norm': True,
            'activation': 'relu',  # ReLU is CPU-efficient
            'use_attention': False,  # Disable attention for CPU efficiency
            'gradient_checkpointing': self.memory_gb < 8,  # Enable for low memory
        }
        
        logger.info(f"CPU-optimized model config: {config}")
        return config
    
    def get_cpu_training_config(self) -> Dict[str, Any]:
        """
        Get CPU-optimized training configuration.
        
        Returns:
            Dictionary with CPU-optimized training parameters
        """
        config = {
            'num_epochs': 30 if self.memory_gb < 4 else 50,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'warmup_epochs': 3,
            'gradient_clip_norm': 1.0,
            'accumulate_grad_batches': 2 if self.memory_gb < 4 else 1,  # Gradient accumulation for small memory
            'early_stopping_patience': 10,
            'validation_frequency': 5,  # Less frequent validation to save time
        }
        
        logger.info(f"CPU-optimized training config: {config}")
        return config


class MemoryMonitor:
    """
    Real-time memory monitoring and automatic resource management.
    
    Features:
    - Continuous memory usage tracking
    - Automatic garbage collection triggers
    - Memory leak detection
    - Resource usage alerts
    """
    
    def __init__(self,
                 memory_threshold: float = 0.85,
                 check_interval: float = 30.0,
                 error_handler: Optional[ErrorHandler] = None):
        """
        Initialize memory monitor.
        
        Args:
            memory_threshold: Memory usage threshold (0.0-1.0) for triggering cleanup
            check_interval: Interval in seconds between memory checks
            error_handler: Error handler instance
        """
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.error_handler = error_handler or global_error_handler
        
        self.monitoring = False
        self.monitor_thread = None
        self.memory_history = deque(maxlen=100)  # Keep last 100 measurements
        self.cleanup_count = 0
        self.alert_count = 0
        
        logger.info(f"MemoryMonitor initialized: threshold={memory_threshold:.1%}, "
                   f"check_interval={check_interval}s")
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                memory_info = self._get_memory_info()
                self.memory_history.append(memory_info)
                
                # Check for high memory usage
                if memory_info['system_usage'] > self.memory_threshold:
                    self._handle_high_memory_usage(memory_info)
                
                # Check for potential memory leaks
                if len(self.memory_history) >= 10:
                    self._check_memory_trend()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        system_memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        info = {
            'timestamp': time.time(),
            'system_total_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_usage': system_memory.percent / 100.0,
            'process_rss_mb': process_memory.rss / (1024**2),
            'process_vms_mb': process_memory.vms / (1024**2),
        }
        
        # Add GPU memory if available
        if torch.cuda.is_available():
            try:
                info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
                info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
            except:
                info['gpu_allocated_mb'] = 0.0
                info['gpu_reserved_mb'] = 0.0
        
        return info
    
    def _handle_high_memory_usage(self, memory_info: Dict[str, float]):
        """Handle high memory usage situation."""
        self.alert_count += 1
        
        logger.warning(f"High memory usage detected: {memory_info['system_usage']:.1%} "
                      f"(threshold: {self.memory_threshold:.1%})")
        
        # Trigger garbage collection
        self._force_cleanup()
        
        # Log memory info
        logger.info(f"Memory info: System={memory_info['system_available_gb']:.1f}GB available, "
                   f"Process={memory_info['process_rss_mb']:.1f}MB RSS")
    
    def _check_memory_trend(self):
        """Check for concerning memory usage trends."""
        if len(self.memory_history) < 10:
            return
        
        # Get recent memory usage trend
        recent_usage = [info['system_usage'] for info in list(self.memory_history)[-10:]]
        
        # Check for consistently increasing memory usage (potential leak)
        if len(recent_usage) >= 5:
            trend = np.polyfit(range(len(recent_usage)), recent_usage, 1)[0]
            if trend > 0.01:  # Increasing by more than 1% per measurement
                logger.warning(f"Potential memory leak detected: usage trend +{trend:.3f} per check")
    
    def _force_cleanup(self):
        """Force memory cleanup."""
        self.cleanup_count += 1
        
        # Python garbage collection
        collected = gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.debug(f"Memory cleanup performed: {collected} objects collected")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory monitoring statistics."""
        current_info = self._get_memory_info()
        
        stats = {
            'current_memory': current_info,
            'monitoring_active': self.monitoring,
            'cleanup_count': self.cleanup_count,
            'alert_count': self.alert_count,
            'memory_threshold': self.memory_threshold,
            'history_length': len(self.memory_history)
        }
        
        if self.memory_history:
            usage_history = [info['system_usage'] for info in self.memory_history]
            stats.update({
                'avg_memory_usage': np.mean(usage_history),
                'max_memory_usage': np.max(usage_history),
                'min_memory_usage': np.min(usage_history)
            })
        
        return stats
    
    @contextmanager
    def memory_tracking_context(self, operation_name: str = "operation"):
        """Context manager for tracking memory usage during operations."""
        start_info = self._get_memory_info()
        start_time = time.time()
        
        logger.debug(f"Starting memory tracking for: {operation_name}")
        
        try:
            yield
        finally:
            end_info = self._get_memory_info()
            end_time = time.time()
            
            duration = end_time - start_time
            memory_delta = end_info['process_rss_mb'] - start_info['process_rss_mb']
            
            logger.info(f"Memory tracking for {operation_name}: "
                       f"duration={duration:.2f}s, memory_delta={memory_delta:+.1f}MB")


class ProgressTracker:
    """
    Comprehensive progress tracking and logging for long-running processes.
    
    Features:
    - Multi-level progress tracking (epochs, batches, etc.)
    - ETA estimation
    - Performance metrics collection
    - Detailed logging with configurable verbosity
    """
    
    def __init__(self,
                 name: str = "Process",
                 log_interval: int = 10,
                 enable_eta: bool = True):
        """
        Initialize progress tracker.
        
        Args:
            name: Name of the process being tracked
            log_interval: Interval for progress logging
            enable_eta: Whether to calculate and display ETA
        """
        self.name = name
        self.log_interval = log_interval
        self.enable_eta = enable_eta
        
        self.start_time = None
        self.current_step = 0
        self.total_steps = None
        self.step_times = deque(maxlen=100)  # For ETA calculation
        self.metrics_history = []
        
        # Nested progress tracking
        self.sub_trackers = {}
        self.current_sub_tracker = None
        
        logger.info(f"ProgressTracker initialized: {name}")
    
    def start(self, total_steps: Optional[int] = None):
        """Start progress tracking."""
        self.start_time = time.time()
        self.total_steps = total_steps
        self.current_step = 0
        self.step_times.clear()
        self.metrics_history.clear()
        
        logger.info(f"Started {self.name}" + 
                   (f" ({total_steps} steps)" if total_steps else ""))
    
    def update(self, step: Optional[int] = None, metrics: Optional[Dict[str, float]] = None):
        """
        Update progress.
        
        Args:
            step: Current step number (auto-increments if None)
            metrics: Optional metrics to log
        """
        current_time = time.time()
        
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Record step time for ETA calculation
        if self.start_time:
            self.step_times.append(current_time)
        
        # Record metrics
        if metrics:
            metrics_entry = {
                'step': self.current_step,
                'timestamp': current_time,
                **metrics
            }
            self.metrics_history.append(metrics_entry)
        
        # Log progress at intervals
        if self.current_step % self.log_interval == 0 or self.current_step == self.total_steps:
            self._log_progress(metrics)
    
    def _log_progress(self, metrics: Optional[Dict[str, float]] = None):
        """Log current progress."""
        if not self.start_time:
            return
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Build progress message
        progress_msg = f"{self.name} - Step {self.current_step}"
        
        if self.total_steps:
            progress_pct = (self.current_step / self.total_steps) * 100
            progress_msg += f"/{self.total_steps} ({progress_pct:.1f}%)"
        
        progress_msg += f" - Elapsed: {self._format_duration(elapsed)}"
        
        # Add ETA if enabled and possible
        if self.enable_eta and self.total_steps and len(self.step_times) >= 2:
            eta = self._calculate_eta()
            if eta:
                progress_msg += f" - ETA: {self._format_duration(eta)}"
        
        # Add metrics
        if metrics:
            metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                   for k, v in metrics.items()])
            progress_msg += f" - {metrics_str}"
        
        # Add performance info
        if len(self.step_times) >= 2:
            recent_times = list(self.step_times)[-10:]  # Last 10 steps
            if len(recent_times) >= 2:
                step_duration = (recent_times[-1] - recent_times[0]) / (len(recent_times) - 1)
                steps_per_sec = 1.0 / step_duration if step_duration > 0 else 0
                progress_msg += f" - {steps_per_sec:.2f} steps/sec"
        
        logger.info(progress_msg)
    
    def _calculate_eta(self) -> Optional[float]:
        """Calculate estimated time to completion."""
        if not self.total_steps or len(self.step_times) < 2:
            return None
        
        # Use recent step times for better ETA accuracy
        recent_times = list(self.step_times)[-20:]  # Last 20 steps
        if len(recent_times) < 2:
            return None
        
        # Calculate average step duration
        total_duration = recent_times[-1] - recent_times[0]
        avg_step_duration = total_duration / (len(recent_times) - 1)
        
        # Calculate remaining steps and ETA
        remaining_steps = self.total_steps - self.current_step
        eta = remaining_steps * avg_step_duration
        
        return eta if eta > 0 else None
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def add_sub_tracker(self, name: str, total_steps: Optional[int] = None) -> 'ProgressTracker':
        """Add a sub-tracker for nested progress tracking."""
        sub_tracker = ProgressTracker(
            name=f"{self.name}/{name}",
            log_interval=max(1, self.log_interval // 5),  # More frequent logging for sub-tasks
            enable_eta=self.enable_eta
        )
        
        self.sub_trackers[name] = sub_tracker
        return sub_tracker
    
    def finish(self, final_metrics: Optional[Dict[str, float]] = None):
        """Finish progress tracking."""
        if not self.start_time:
            return
        
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        # Final progress log
        final_msg = f"{self.name} completed in {self._format_duration(total_duration)}"
        
        if self.total_steps:
            steps_per_sec = self.current_step / total_duration if total_duration > 0 else 0
            final_msg += f" - {self.current_step} steps ({steps_per_sec:.2f} steps/sec)"
        
        if final_metrics:
            metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                   for k, v in final_metrics.items()])
            final_msg += f" - {metrics_str}"
        
        logger.info(final_msg)
        
        # Record final metrics
        if final_metrics:
            final_entry = {
                'step': self.current_step,
                'timestamp': end_time,
                'total_duration': total_duration,
                **final_metrics
            }
            self.metrics_history.append(final_entry)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress tracking summary."""
        if not self.start_time:
            return {'status': 'not_started'}
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        summary = {
            'name': self.name,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'elapsed_time': elapsed,
            'start_time': self.start_time,
            'metrics_count': len(self.metrics_history),
            'sub_trackers': list(self.sub_trackers.keys())
        }
        
        if self.total_steps:
            summary['progress_ratio'] = self.current_step / self.total_steps
            summary['estimated_total_time'] = elapsed / (self.current_step / self.total_steps) if self.current_step > 0 else None
        
        if len(self.step_times) >= 2:
            recent_times = list(self.step_times)[-10:]
            if len(recent_times) >= 2:
                step_duration = (recent_times[-1] - recent_times[0]) / (len(recent_times) - 1)
                summary['steps_per_second'] = 1.0 / step_duration if step_duration > 0 else 0
        
        return summary


class PerformanceOptimizer:
    """
    Main performance optimization coordinator that integrates all optimization components.
    
    Features:
    - Automatic configuration optimization based on system resources
    - Integrated caching, monitoring, and progress tracking
    - Performance profiling and recommendations
    - Resource usage optimization
    """
    
    def __init__(self,
                 cache_dir: Union[str, Path] = "cache",
                 enable_monitoring: bool = True,
                 error_handler: Optional[ErrorHandler] = None):
        """
        Initialize performance optimizer.
        
        Args:
            cache_dir: Directory for caching
            enable_monitoring: Whether to enable memory monitoring
            error_handler: Error handler instance
        """
        self.error_handler = error_handler or global_error_handler
        
        # Initialize components
        self.graph_cache = GraphCache(
            cache_dir=Path(cache_dir) / "graphs",
            error_handler=self.error_handler
        )
        
        self.cpu_optimizer = CPUOptimizer(error_handler=self.error_handler)
        
        # Add device manager
        from polymer_prediction.utils.error_handling import DeviceManager
        self.device_manager = DeviceManager(error_handler=self.error_handler)
        
        self.memory_monitor = MemoryMonitor(error_handler=self.error_handler)
        
        if enable_monitoring:
            self.memory_monitor.start_monitoring()
        
        # Performance tracking
        self.performance_history = []
        
        logger.info("PerformanceOptimizer initialized with all components")
    
    def get_optimized_config(self, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get optimized configuration based on system resources.
        
        Args:
            base_config: Base configuration to optimize
            
        Returns:
            Optimized configuration dictionary
        """
        if base_config is None:
            base_config = {}
        
        # Get CPU optimizations
        cpu_model_config = self.cpu_optimizer.get_cpu_model_config()
        cpu_training_config = self.cpu_optimizer.get_cpu_training_config()
        
        # Merge configurations
        optimized_config = {
            **base_config,
            'model': {**base_config.get('model', {}), **cpu_model_config},
            'training': {**base_config.get('training', {}), **cpu_training_config},
            'data': {
                **base_config.get('data', {}),
                'batch_size': self.cpu_optimizer.get_optimal_batch_size(
                    base_config.get('data', {}).get('batch_size', 32)
                ),
                'num_workers': self.cpu_optimizer.get_optimal_num_workers(),
                'pin_memory': False,  # Disable for CPU training
                'persistent_workers': False
            },
            'caching': {
                'enable_graph_cache': True,
                'cache_dir': str(self.graph_cache.cache_dir.parent),
                'max_memory_items': self.graph_cache.max_memory_items,
                'max_disk_size_gb': self.graph_cache.max_disk_size_gb
            },
            'monitoring': {
                'enable_memory_monitoring': self.memory_monitor.monitoring,
                'memory_threshold': self.memory_monitor.memory_threshold,
                'check_interval': self.memory_monitor.check_interval
            }
        }
        
        logger.info("Generated optimized configuration based on system resources")
        return optimized_config
    
    @contextmanager
    def performance_context(self, operation_name: str = "operation"):
        """
        Context manager for comprehensive performance tracking.
        
        Args:
            operation_name: Name of the operation being tracked
        """
        # Start performance tracking
        start_time = time.time()
        start_metrics = PerformanceMetrics(start_time=start_time)
        
        # Get initial system state
        initial_memory = self.memory_monitor._get_memory_info()
        initial_cache_stats = self.graph_cache.get_stats()
        
        logger.info(f"Starting performance tracking for: {operation_name}")
        
        try:
            with self.memory_monitor.memory_tracking_context(operation_name):
                yield {
                    'graph_cache': self.graph_cache,
                    'memory_monitor': self.memory_monitor,
                    'cpu_optimizer': self.cpu_optimizer,
                    'start_metrics': start_metrics
                }
        finally:
            # Finalize performance tracking
            end_time = time.time()
            start_metrics.end_time = end_time
            
            # Get final system state
            final_memory = self.memory_monitor._get_memory_info()
            final_cache_stats = self.graph_cache.get_stats()
            
            # Calculate performance metrics
            start_metrics.memory_usage_mb = final_memory['process_rss_mb']
            start_metrics.peak_memory_mb = max(initial_memory['process_rss_mb'], final_memory['process_rss_mb'])
            start_metrics.cache_hits = final_cache_stats['memory_hits'] + final_cache_stats['disk_hits'] - \
                                     (initial_cache_stats['memory_hits'] + initial_cache_stats['disk_hits'])
            start_metrics.cache_misses = final_cache_stats['misses'] - initial_cache_stats['misses']
            
            start_metrics.finalize()
            
            # Record performance history
            self.performance_history.append({
                'operation': operation_name,
                'metrics': asdict(start_metrics),
                'timestamp': end_time
            })
            
            # Log performance summary
            self._log_performance_summary(operation_name, start_metrics)
    
    def _log_performance_summary(self, operation_name: str, metrics: PerformanceMetrics):
        """Log performance summary."""
        summary_msg = f"Performance summary for {operation_name}:"
        summary_msg += f" duration={metrics.duration:.2f}s"
        
        if metrics.samples_processed > 0:
            summary_msg += f", samples={metrics.samples_processed}"
            if metrics.samples_per_second:
                summary_msg += f" ({metrics.samples_per_second:.1f}/sec)"
        
        if metrics.memory_usage_mb:
            summary_msg += f", memory={metrics.memory_usage_mb:.1f}MB"
        
        if metrics.cache_hit_ratio is not None:
            summary_msg += f", cache_hit_ratio={metrics.cache_hit_ratio:.2%}"
        
        logger.info(summary_msg)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'system_info': {
                'cpu_count': self.cpu_optimizer.cpu_count,
                'memory_gb': self.cpu_optimizer.memory_gb,
                'cuda_available': torch.cuda.is_available()
            },
            'cache_stats': self.graph_cache.get_stats(),
            'memory_stats': self.memory_monitor.get_memory_stats(),
            'performance_history': self.performance_history[-10:],  # Last 10 operations
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Cache recommendations
        cache_stats = self.graph_cache.get_stats()
        if cache_stats['hit_ratio'] < 0.5:
            recommendations.append("Consider increasing graph cache size for better performance")
        
        # Memory recommendations
        memory_stats = self.memory_monitor.get_memory_stats()
        if memory_stats['alert_count'] > 0:
            recommendations.append("High memory usage detected - consider reducing batch size")
        
        # CPU recommendations
        if self.cpu_optimizer.memory_gb < 8:
            recommendations.append("Limited RAM detected - enable gradient accumulation for larger effective batch sizes")
        
        if self.cpu_optimizer.cpu_count <= 2:
            recommendations.append("Limited CPU cores - consider using single-threaded data loading")
        
        return recommendations
    
    def cleanup(self):
        """Clean up resources."""
        self.memory_monitor.stop_monitoring()
        logger.info("PerformanceOptimizer cleanup completed")