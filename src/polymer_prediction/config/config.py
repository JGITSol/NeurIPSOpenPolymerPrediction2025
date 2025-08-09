"""Configuration settings for the polymer prediction project."""

import os
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """Path configuration for cross-platform compatibility."""
    
    # Base directories
    data_dir: str = "info"
    model_save_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    checkpoints_dir: str = "checkpoints"
    outputs_dir: str = "outputs"
    
    # Data files
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    sample_submission_file: str = "sample_submission.csv"
    
    def __post_init__(self):
        """Convert all paths to Path objects for cross-platform compatibility."""
        self.data_path = Path(self.data_dir)
        self.model_save_path = Path(self.model_save_dir)
        self.results_path = Path(self.results_dir)
        self.logs_path = Path(self.logs_dir)
        self.cache_path = Path(self.cache_dir)
        self.checkpoints_path = Path(self.checkpoints_dir)
        self.outputs_path = Path(self.outputs_dir)
        
        # Full file paths
        self.train_path = self.data_path / self.train_file
        self.test_path = self.data_path / self.test_file
        self.sample_submission_path = self.data_path / self.sample_submission_file
    
    def create_directories(self):
        """Create all necessary directories."""
        directories = [
            self.model_save_path,
            self.results_path,
            self.logs_path,
            self.cache_path,
            self.checkpoints_path,
            self.outputs_path
        ]
        
        for directory in directories:
            if isinstance(directory, str):
                directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model configuration."""
    
    # GCN Model parameters
    hidden_channels: int = 128
    num_gcn_layers: int = 3
    dropout: float = 0.3
    activation: str = "relu"
    batch_norm: bool = True
    
    # Tree ensemble models
    tree_models: List[str] = field(default_factory=lambda: ["lgbm", "xgb", "catboost"])
    
    # Ensemble parameters
    use_stacking: bool = True
    meta_model: str = "ridge"
    n_folds: int = 5


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Basic training parameters
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Optimization parameters
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0
    
    # Cross-validation
    n_folds: int = 5
    random_state: int = 42
    
    # Resource management
    max_retries: int = 3
    min_batch_size: int = 1
    memory_threshold: float = 0.8  # 80% memory usage threshold


@dataclass
class DataConfig:
    """Data configuration."""
    
    # Target columns
    target_cols: List[str] = field(default_factory=lambda: ["Tg", "FFV", "Tc", "Density", "Rg"])
    
    # Data processing
    smiles_column: str = "SMILES"
    id_column: str = "id"
    test_split: float = 0.2
    val_split: float = 0.1
    random_seed: int = 42
    
    # DataLoader parameters
    num_workers: int = 0  # Set to 0 for Windows compatibility
    pin_memory: bool = True
    drop_last: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    log_file: Optional[str] = "polymer_prediction.log"
    use_structured_logging: bool = True
    log_to_console: bool = True
    log_to_file: bool = True
    
    # Performance logging
    log_memory_usage: bool = True
    log_training_progress: bool = True
    log_model_performance: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    
    # Caching
    enable_graph_cache: bool = True
    cache_size_limit: int = 1000  # Maximum number of cached graphs
    
    # Memory management
    enable_memory_monitoring: bool = True
    memory_cleanup_frequency: int = 10  # Clean up every N epochs
    
    # CPU optimization
    enable_cpu_optimization: bool = True
    cpu_batch_size_factor: float = 0.5  # Reduce batch size for CPU
    
    # Progress tracking
    enable_progress_tracking: bool = True
    progress_update_frequency: int = 10  # Update every N batches


class Config:
    """Main configuration class that combines all sub-configurations."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration with optional overrides.
        
        Args:
            config_dict: Optional dictionary to override default values
        """
        # Initialize sub-configurations
        self.paths = PathConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        self.performance = PerformanceConfig()
        
        # Device configuration
        self.device = self._detect_device()
        
        # Apply CPU optimizations if needed
        if self.device.type == "cpu":
            self._apply_cpu_optimizations()
        
        # Apply any overrides
        if config_dict:
            self._apply_overrides(config_dict)
        
        # Create necessary directories
        self.paths.create_directories()
    
    def _detect_device(self) -> torch.device:
        """Detect the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
        
        return device
    
    def _apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations."""
        # Reduce batch size for CPU training
        original_batch_size = self.training.batch_size
        self.training.batch_size = max(1, int(original_batch_size * self.performance.cpu_batch_size_factor))
        
        # Reduce model complexity for CPU
        self.model.hidden_channels = min(self.model.hidden_channels, 64)
        self.model.num_gcn_layers = min(self.model.num_gcn_layers, 2)
        
        # Adjust training parameters
        self.training.num_epochs = min(self.training.num_epochs, 30)
        self.data.num_workers = 0  # Disable multiprocessing on CPU
    
    def _apply_overrides(self, config_dict: Dict[str, Any]):
        """Apply configuration overrides from dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section) and isinstance(values, dict):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "paths": self.paths.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "logging": self.logging.__dict__,
            "performance": self.performance.__dict__,
            "device": str(self.device)
        }
    
    def save_config(self, filepath: str):
        """Save configuration to file."""
        import json
        
        config_dict = self.to_dict()
        # Convert Path objects to strings for JSON serialization
        for section in config_dict.values():
            if isinstance(section, dict):
                for key, value in section.items():
                    if isinstance(value, Path):
                        section[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'Config':
        """Load configuration from file."""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for debugging."""
        import platform
        import sys
        
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device": str(self.device),
            "working_directory": str(Path.cwd())
        }


# Create a default configuration instance
CONFIG = Config()

# Legacy compatibility - expose commonly used attributes at module level
DEVICE = CONFIG.device
NUM_EPOCHS = CONFIG.training.num_epochs
BATCH_SIZE = CONFIG.training.batch_size
LEARNING_RATE = CONFIG.training.learning_rate
HIDDEN_CHANNELS = CONFIG.model.hidden_channels
NUM_GCN_LAYERS = CONFIG.model.num_gcn_layers
TARGET_COLS = CONFIG.data.target_cols
