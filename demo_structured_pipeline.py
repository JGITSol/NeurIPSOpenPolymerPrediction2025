"""
Demonstration of the improved code structure and configuration management.

This script shows the key improvements made in task 8:
1. Centralized configuration management
2. Proper module separation
3. Cross-platform path handling
4. Structured logging
5. Compatible dependencies
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from polymer_prediction.config.config import Config
from polymer_prediction.utils.logging import setup_logging, get_logger, get_performance_timer
from polymer_prediction.utils.path_manager import PathManager
from polymer_prediction.pipeline.main_pipeline import MainPipeline


def demonstrate_configuration():
    """Demonstrate the centralized configuration system."""
    print("=== Configuration Management Demo ===")
    
    # Create default configuration
    config = Config()
    
    print(f"Device detected: {config.device}")
    print(f"Training epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Model hidden channels: {config.model.hidden_channels}")
    
    # Show CPU optimizations
    if config.device.type == "cpu":
        print("CPU optimizations applied:")
        print(f"  - Reduced batch size: {config.training.batch_size}")
        print(f"  - Reduced hidden channels: {config.model.hidden_channels}")
    
    # Demonstrate configuration overrides
    overrides = {
        "training": {"num_epochs": 25, "batch_size": 8},
        "model": {"hidden_channels": 64}
    }
    
    config_with_overrides = Config(overrides)
    print(f"\nWith overrides:")
    print(f"  - Epochs: {config_with_overrides.training.num_epochs}")
    print(f"  - Batch size: {config_with_overrides.training.batch_size}")
    print(f"  - Hidden channels: {config_with_overrides.model.hidden_channels}")
    
    return config


def demonstrate_path_management():
    """Demonstrate cross-platform path management."""
    print("\n=== Path Management Demo ===")
    
    path_manager = PathManager()
    
    print(f"Base path: {path_manager.base_path}")
    print(f"Platform: {path_manager.platform}")
    
    # Show different path types
    paths = {
        "Data file": path_manager.get_data_path("train.csv"),
        "Model file": path_manager.get_model_path("best_model.pt"),
        "Output file": path_manager.get_output_path("submission.csv"),
        "Log file": path_manager.get_log_path("pipeline.log"),
        "Cache file": path_manager.get_cache_path("graphs.pkl")
    }
    
    for path_type, path in paths.items():
        print(f"  {path_type}: {path}")
    
    # Demonstrate directory creation
    test_dir = path_manager.ensure_directory("demo_output")
    print(f"Created directory: {test_dir}")
    
    return path_manager


def demonstrate_structured_logging():
    """Demonstrate structured logging capabilities."""
    print("\n=== Structured Logging Demo ===")
    
    # Setup logging
    setup_logging(
        log_level="INFO",
        log_file="demo_structured.log",
        enable_structured_logging=True,
        log_to_console=True,
        log_to_file=True
    )
    
    logger = get_logger("demo")
    
    # Basic logging
    logger.info("This is a basic info message")
    
    # Logging with context
    logger.add_context(demo_mode=True, version="1.0")
    logger.info("This message includes context")
    
    # Performance logging
    import time
    with get_performance_timer(logger, "demo_operation", operation_type="test"):
        time.sleep(0.1)  # Simulate work
    
    # Memory logging
    try:
        import psutil
        process = psutil.Process()
        memory_info = {
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "cpu_percent": process.cpu_percent()
        }
        logger.log_memory_usage("demo_stage", memory_info)
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
    
    # Model metrics logging
    demo_metrics = {
        "accuracy": 0.95,
        "loss": 0.123,
        "f1_score": 0.89
    }
    logger.log_model_metrics("demo_model", demo_metrics)
    
    # Data statistics logging
    demo_stats = {
        "num_samples": 1000,
        "num_features": 50,
        "missing_values": 25
    }
    logger.log_data_statistics("preprocessing", demo_stats)
    
    logger.clear_context()
    
    return logger


def demonstrate_pipeline_structure():
    """Demonstrate the modular pipeline structure."""
    print("\n=== Pipeline Structure Demo ===")
    
    config = Config()
    path_manager = PathManager()
    
    # Create main pipeline
    pipeline = MainPipeline(config, path_manager)
    
    print("Pipeline components:")
    print(f"  Data pipeline: {type(pipeline.data_pipeline).__name__}")
    print(f"  Training pipeline: {type(pipeline.training_pipeline).__name__}")
    print(f"  Prediction pipeline: {type(pipeline.prediction_pipeline).__name__}")
    
    # Show configuration integration
    print(f"\nConfiguration integration:")
    print(f"  All components use same config: {id(pipeline.config)}")
    print(f"  Data pipeline config: {id(pipeline.data_pipeline.config)}")
    print(f"  Training pipeline config: {id(pipeline.training_pipeline.config)}")
    print(f"  Prediction pipeline config: {id(pipeline.prediction_pipeline.config)}")
    
    return pipeline


def demonstrate_environment_info():
    """Demonstrate environment information gathering."""
    print("\n=== Environment Information Demo ===")
    
    config = Config()
    env_info = config.get_environment_info()
    
    for key, value in env_info.items():
        print(f"  {key}: {value}")


def main():
    """Run all demonstrations."""
    print("Polymer Prediction - Improved Code Structure Demo")
    print("=" * 60)
    
    try:
        # Demonstrate each improvement
        config = demonstrate_configuration()
        path_manager = demonstrate_path_management()
        logger = demonstrate_structured_logging()
        pipeline = demonstrate_pipeline_structure()
        demonstrate_environment_info()
        
        print("\n=== Summary ===")
        print("✓ Centralized configuration management")
        print("✓ Cross-platform path handling")
        print("✓ Structured logging with context")
        print("✓ Modular pipeline architecture")
        print("✓ Environment detection and adaptation")
        print("✓ Compatible dependency management")
        
        logger.info("Demo completed successfully")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()