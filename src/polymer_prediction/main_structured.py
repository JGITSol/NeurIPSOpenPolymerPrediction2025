"""
Structured main script for polymer prediction using the new modular pipeline.

This script demonstrates the improved code structure with proper separation of concerns,
centralized configuration management, cross-platform path handling, and structured logging.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from polymer_prediction.config.config import Config
from polymer_prediction.pipeline.main_pipeline import MainPipeline, run_polymer_prediction_pipeline
from polymer_prediction.utils.logging import setup_logging, get_logger, get_performance_timer
from polymer_prediction.utils.path_manager import PathManager

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Structured Polymer Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["gcn", "tree_ensemble", "stacking_ensemble"],
        default="stacking_ensemble",
        help="Type of model to use for final predictions"
    )
    
    # Data options
    parser.add_argument(
        "--data-dir",
        type=str,
        default="info",
        help="Directory containing data files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output files"
    )
    
    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate"
    )
    
    # Logging options
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file name"
    )
    
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging"
    )
    
    # Performance options
    parser.add_argument(
        "--disable-gpu",
        action="store_true",
        help="Force CPU-only training"
    )
    
    parser.add_argument(
        "--enable-caching",
        action="store_true",
        default=True,
        help="Enable graph caching for performance"
    )
    
    # Development options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save the effective configuration to file"
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> Config:
    """Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration object
    """
    # Load base configuration
    if args.config:
        config = Config.load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = Config()
        logger.info("Using default configuration")
    
    # Apply command line overrides
    overrides = {}
    
    # Path overrides
    if args.data_dir:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    
    if args.output_dir:
        overrides.setdefault("paths", {})["outputs_dir"] = args.output_dir
    
    # Training overrides
    if args.epochs:
        overrides.setdefault("training", {})["num_epochs"] = args.epochs
    
    if args.batch_size:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    
    if args.learning_rate:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    
    # Logging overrides
    if args.log_level:
        overrides.setdefault("logging", {})["level"] = args.log_level
    
    if args.log_file:
        overrides.setdefault("logging", {})["log_file"] = args.log_file
    
    if args.no_console_log:
        overrides.setdefault("logging", {})["log_to_console"] = False
    
    # Performance overrides
    if args.disable_gpu:
        # Force CPU device
        import torch
        config.device = torch.device("cpu")
        config._apply_cpu_optimizations()
    
    if args.enable_caching:
        overrides.setdefault("performance", {})["enable_graph_cache"] = True
    
    # Debug mode
    if args.debug:
        overrides.setdefault("logging", {})["level"] = "DEBUG"
        overrides.setdefault("logging", {})["log_to_console"] = True
    
    # Apply overrides
    if overrides:
        config._apply_overrides(overrides)
        logger.info("Applied command line configuration overrides")
    
    return config


def setup_environment(config: Config, args):
    """Setup the execution environment.
    
    Args:
        config: Configuration object
        args: Command line arguments
    """
    # Setup logging
    setup_logging(
        log_level=config.logging.level,
        log_file=config.logging.log_file if config.logging.log_to_file else None,
        log_dir=str(config.paths.logs_path),
        enable_structured_logging=config.logging.use_structured_logging,
        log_to_console=config.logging.log_to_console,
        log_to_file=config.logging.log_to_file
    )
    
    logger.info("=== Structured Polymer Prediction Pipeline ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    
    # Log configuration if in debug mode
    if args.debug:
        logger.debug("Configuration details:")
        config_dict = config.to_dict()
        for section, values in config_dict.items():
            logger.debug(f"  {section}: {values}")
    
    # Save configuration if requested
    if args.save_config:
        path_manager = PathManager()
        config_path = path_manager.get_output_path("effective_config.json")
        config.save_config(str(config_path))
        logger.info(f"Effective configuration saved to {config_path}")


def main():
    """Main entry point for the structured polymer prediction pipeline."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create configuration
        config = create_config_from_args(args)
        
        # Setup environment
        setup_environment(config, args)
        
        # Log startup information
        logger.info("Starting polymer prediction pipeline...")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Data directory: {config.paths.data_path}")
        logger.info(f"Output directory: {config.paths.outputs_path}")
        
        # Create and run pipeline
        pipeline = MainPipeline(config)
        
        with get_performance_timer(logger, "complete_pipeline"):
            results = pipeline.run_complete_pipeline(args.model_type)
        
        # Report results
        if results["success"]:
            logger.info("=== Pipeline Completed Successfully ===")
            
            # Log key results
            if "prediction_generation" in results["results"]:
                pred_results = results["results"]["prediction_generation"]
                if "prediction_results" in pred_results:
                    submission_path = pred_results["prediction_results"].get("submission_path")
                    if submission_path:
                        logger.info(f"Submission file saved to: {submission_path}")
                    
                    pred_summary = pred_results["prediction_results"].get("prediction_summary")
                    if pred_summary:
                        logger.info(f"Generated predictions for {pred_summary['num_samples']} samples")
            
            if results.get("report_path"):
                logger.info(f"Detailed report saved to: {results['report_path']}")
            
            return 0
        else:
            logger.error("=== Pipeline Failed ===")
            logger.error(f"Error: {results.get('error', 'Unknown error')}")
            
            if results.get("report_path"):
                logger.error(f"Failure report saved to: {results['report_path']}")
            
            return 1
    
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


def run_with_config(config_path: str, model_type: str = "stacking_ensemble"):
    """Convenience function to run pipeline with a configuration file.
    
    Args:
        config_path: Path to configuration file
        model_type: Type of model to use for predictions
        
    Returns:
        Pipeline results
    """
    return run_polymer_prediction_pipeline(config_path, model_type)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)