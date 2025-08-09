"""Tests for the structured pipeline architecture."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from polymer_prediction.config.config import Config, PathConfig, ModelConfig, TrainingConfig
from polymer_prediction.utils.path_manager import PathManager
from polymer_prediction.utils.logging import get_logger, StructuredLogger
from polymer_prediction.pipeline.data_pipeline import DataPipeline
from polymer_prediction.pipeline.training_pipeline import TrainingPipeline
from polymer_prediction.pipeline.prediction_pipeline import PredictionPipeline
from polymer_prediction.pipeline.main_pipeline import MainPipeline


class TestConfig:
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test that configuration initializes correctly."""
        config = Config()
        
        assert config.device is not None
        assert config.paths is not None
        assert config.model is not None
        assert config.training is not None
        assert config.data is not None
        assert config.logging is not None
        assert config.performance is not None
    
    def test_config_overrides(self):
        """Test configuration overrides."""
        overrides = {
            "training": {"num_epochs": 100, "batch_size": 64},
            "model": {"hidden_channels": 256}
        }
        
        config = Config(overrides)
        
        assert config.training.num_epochs == 100
        assert config.training.batch_size == 64
        assert config.model.hidden_channels == 256
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config = Config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save_config(f.name)
            
            # Load config back
            loaded_config = Config.load_config(f.name)
            
            assert loaded_config.training.num_epochs == config.training.num_epochs
            assert loaded_config.model.hidden_channels == config.model.hidden_channels
    
    def test_path_config(self):
        """Test path configuration."""
        path_config = PathConfig()
        
        assert isinstance(path_config.data_path, Path)
        assert isinstance(path_config.model_save_path, Path)
        assert isinstance(path_config.train_path, Path)
        assert isinstance(path_config.test_path, Path)


class TestPathManager:
    """Test path management utilities."""
    
    def test_path_manager_initialization(self):
        """Test path manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager = PathManager(temp_dir)
            
            assert path_manager.base_path == Path(temp_dir).resolve()
    
    def test_path_resolution(self):
        """Test path resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager = PathManager(temp_dir)
            
            # Test relative path resolution
            relative_path = path_manager.resolve_path("test/file.txt")
            expected_path = Path(temp_dir) / "test" / "file.txt"
            
            assert relative_path.resolve() == expected_path.resolve()
    
    def test_directory_creation(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager = PathManager(temp_dir)
            
            test_dir = path_manager.ensure_directory("test_dir")
            
            assert test_dir.exists()
            assert test_dir.is_dir()
    
    def test_file_operations(self):
        """Test file operation utilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager = PathManager(temp_dir)
            
            # Create a test file
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")
            
            assert path_manager.file_exists("test.txt")
            assert path_manager.get_file_size("test.txt") > 0
            assert not path_manager.directory_exists("test.txt")


class TestStructuredLogger:
    """Test structured logging functionality."""
    
    def test_logger_creation(self):
        """Test logger creation."""
        logger = get_logger("test_logger")
        
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "test_logger"
    
    def test_logger_context(self):
        """Test logger context management."""
        logger = get_logger("test_logger")
        
        logger.add_context(test_key="test_value")
        assert logger._context["test_key"] == "test_value"
        
        logger.clear_context()
        assert len(logger._context) == 0
    
    def test_performance_logging(self):
        """Test performance logging."""
        logger = get_logger("test_logger")
        
        # This should not raise an exception
        logger.log_performance("test_operation", 1.23, extra_metric=456)
    
    def test_memory_logging(self):
        """Test memory usage logging."""
        logger = get_logger("test_logger")
        
        memory_info = {"memory_mb": 100, "cpu_percent": 50}
        
        # This should not raise an exception
        logger.log_memory_usage("test_stage", memory_info)


class TestPipelineComponents:
    """Test individual pipeline components."""
    
    def test_data_pipeline_initialization(self):
        """Test data pipeline initialization."""
        config = Config()
        data_pipeline = DataPipeline(config)
        
        assert data_pipeline.config == config
        assert data_pipeline.path_manager is not None
        assert data_pipeline.error_handler is not None
    
    def test_training_pipeline_initialization(self):
        """Test training pipeline initialization."""
        config = Config()
        training_pipeline = TrainingPipeline(config)
        
        assert training_pipeline.config == config
        assert training_pipeline.path_manager is not None
        assert training_pipeline.error_handler is not None
        assert training_pipeline.models == {}
        assert training_pipeline.training_history == {}
    
    def test_prediction_pipeline_initialization(self):
        """Test prediction pipeline initialization."""
        config = Config()
        prediction_pipeline = PredictionPipeline(config)
        
        assert prediction_pipeline.config == config
        assert prediction_pipeline.path_manager is not None
        assert prediction_pipeline.error_handler is not None
    
    def test_main_pipeline_initialization(self):
        """Test main pipeline initialization."""
        config = Config()
        main_pipeline = MainPipeline(config)
        
        assert main_pipeline.config == config
        assert main_pipeline.path_manager is not None
        assert main_pipeline.data_pipeline is not None
        assert main_pipeline.training_pipeline is not None
        assert main_pipeline.prediction_pipeline is not None


class TestIntegration:
    """Test integration between components."""
    
    def test_config_path_manager_integration(self):
        """Test integration between config and path manager."""
        config = Config()
        path_manager = PathManager()
        
        # Test that paths from config work with path manager
        data_path = path_manager.resolve_path(config.paths.data_dir)
        model_path = path_manager.resolve_path(config.paths.model_save_dir)
        
        assert isinstance(data_path, Path)
        assert isinstance(model_path, Path)
    
    def test_pipeline_component_integration(self):
        """Test that pipeline components work together."""
        config = Config()
        path_manager = PathManager()
        
        # Create all pipeline components
        data_pipeline = DataPipeline(config, path_manager)
        training_pipeline = TrainingPipeline(config, path_manager)
        prediction_pipeline = PredictionPipeline(config, path_manager)
        
        # Test that they all use the same configuration
        assert data_pipeline.config == config
        assert training_pipeline.config == config
        assert prediction_pipeline.config == config
        
        # Test that they all use the same path manager
        assert data_pipeline.path_manager == path_manager
        assert training_pipeline.path_manager == path_manager
        assert prediction_pipeline.path_manager == path_manager


if __name__ == "__main__":
    pytest.main([__file__])