"""
Example demonstrating the new structured pipeline architecture.

This example shows how to use the improved code structure with:
- Centralized configuration management
- Proper module separation
- Cross-platform path handling
- Structured logging
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from polymer_prediction.config.config import Config
from polymer_prediction.pipeline.main_pipeline import MainPipeline
from polymer_prediction.utils.logging import setup_logging, get_logger
from polymer_prediction.utils.path_manager import PathManager


def main():
    """Demonstrate the structured pipeline."""
    
    # 1. Create configuration with custom settings
    config = Config()
    
    # Override some settings for this example
    config.training.num_epochs = 10  # Reduced for quick demo
    config.training.batch_size = 16
    config.logging.level = "INFO"
    config.logging.log_file = "structured_example.log"
    
    # 2. Setup logging
    setup_logging(
        log_level=config.logging.level,
        log_file=str(config.paths.logs_path / config.logging.log_file),
        enable_structured_logging=True
    )
    
    logger = get_logger(__name__)
    
    # 3. Log configuration and environment info
    logger.info("=== Structured Pipeline Example ===")
    logger.log_configuration(config.to_dict())
    
    env_info = config.get_environment_info()
    logger.info("Environment information", **env_info)
    
    # 4. Create path manager and demonstrate cross-platform paths
    path_manager = PathManager()
    
    logger.info("Path examples:")
    logger.info(f"Data path: {path_manager.get_data_path('train.csv')}")
    logger.info(f"Model path: {path_manager.get_model_path('example_model.pt')}")
    logger.info(f"Output path: {path_manager.get_output_path('example_submission.csv')}")
    
    # 5. Create and configure pipeline
    pipeline = MainPipeline(config, path_manager)
    
    # 6. Demonstrate structured logging with context
    logger.add_context(example_run=True, demo_mode=True)
    
    try:
        # This would normally run the complete pipeline
        # For this example, we'll just demonstrate the structure
        logger.info("Pipeline structure demonstration:")
        logger.info(f"Data pipeline: {type(pipeline.data_pipeline).__name__}")
        logger.info(f"Training pipeline: {type(pipeline.training_pipeline).__name__}")
        logger.info(f"Prediction pipeline: {type(pipeline.prediction_pipeline).__name__}")
        
        # Demonstrate performance logging
        import time
        from polymer_prediction.utils.logging import get_performance_timer
        with get_performance_timer(logger, "example_operation"):
            time.sleep(0.1)  # Simulate some work
        
        # Demonstrate memory logging
        import psutil
        process = psutil.Process()
        memory_info = {
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent()
        }
        logger.log_memory_usage("example_stage", memory_info)
        
        # Demonstrate model metrics logging
        example_metrics = {
            "train_loss": 0.123,
            "val_loss": 0.145,
            "accuracy": 0.89
        }
        logger.log_model_metrics("example_model", example_metrics)
        
        logger.info("Structured pipeline example completed successfully")
        
    except Exception as e:
        logger.log_error_with_traceback("Example failed", e)
        raise
    
    finally:
        logger.clear_context()


if __name__ == "__main__":
    main()