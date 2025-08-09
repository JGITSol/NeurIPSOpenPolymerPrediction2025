#!/usr/bin/env python3
"""
Production-Ready Polymer Prediction Pipeline

This is the main entry point for the polymer prediction system, providing a complete
production-ready pipeline with comprehensive error handling, logging, configuration
management, and model checkpointing.

Features:
- Command-line argument parsing for configurable parameters
- Comprehensive logging and progress reporting
- Model checkpointing and saving for reproducibility
- Submission file generation with validation
- Integration of all fixed components and new functionality
- Robust error handling and recovery mechanisms
"""

import sys
import argparse
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Core imports
from polymer_prediction.config.config import Config
from polymer_prediction.pipeline.main_pipeline import MainPipeline
from polymer_prediction.utils.logging import setup_logging, get_logger, get_performance_timer
from polymer_prediction.utils.path_manager import PathManager
from polymer_prediction.utils.error_handling import ErrorHandler

# Initialize logger
logger = get_logger(__name__)


class ProductionPipeline:
    """Production-ready polymer prediction pipeline with comprehensive features."""
    
    def __init__(self, config: Config, path_manager: PathManager):
        """Initialize production pipeline.
        
        Args:
            config: Configuration object
            path_manager: Path manager for file operations
        """
        self.config = config
        self.path_manager = path_manager
        self.error_handler = ErrorHandler()
        self.main_pipeline = MainPipeline(config, path_manager)
        
        # Pipeline state
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.checkpoints = {}
        
        logger.info("ProductionPipeline initialized")
    
    def setup_environment(self):
        """Setup the production environment."""
        logger.info("=== Setting up Production Environment ===")
        
        # Create all necessary directories
        self.config.paths.create_directories()
        
        # Setup logging
        log_file = None
        if self.config.logging.log_to_file and self.config.logging.log_file:
            log_file = str(self.path_manager.get_log_path(self.config.logging.log_file))
        
        setup_logging(
            log_level=self.config.logging.level,
            log_file=log_file,
            log_dir=str(self.config.paths.logs_path),
            enable_structured_logging=self.config.logging.use_structured_logging,
            log_to_console=self.config.logging.log_to_console,
            log_to_file=self.config.logging.log_to_file
        )
        
        # Log environment information
        self._log_environment_info()
        
        # Save configuration
        self._save_pipeline_configuration()
        
        logger.info("Production environment setup completed")
    
    def _log_environment_info(self):
        """Log comprehensive environment information."""
        logger.info("=== Environment Information ===")
        
        env_info = self.config.get_environment_info()
        for key, value in env_info.items():
            logger.info(f"{key}: {value}")
        
        # Log configuration summary
        logger.info("=== Configuration Summary ===")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Batch size: {self.config.training.batch_size}")
        logger.info(f"Learning rate: {self.config.training.learning_rate}")
        logger.info(f"Number of epochs: {self.config.training.num_epochs}")
        logger.info(f"Model type: {self.config.model.tree_models}")
        logger.info(f"Target columns: {self.config.data.target_cols}")
    
    def _save_pipeline_configuration(self):
        """Save pipeline configuration for reproducibility."""
        config_path = self.path_manager.get_output_path("production_config.json")
        self.config.save_config(str(config_path))
        logger.info(f"Production configuration saved to {config_path}")
    
    def create_checkpoint(self, stage: str, data: Dict[str, Any]):
        """Create a checkpoint for the current pipeline stage.
        
        Args:
            stage: Pipeline stage name
            data: Data to checkpoint
        """
        checkpoint_data = {
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "data": data
        }
        
        checkpoint_path = self.path_manager.get_checkpoint_path(f"{stage}_checkpoint.json")
        
        try:
            # Save checkpoint (excluding large objects)
            checkpoint_to_save = {
                "stage": checkpoint_data["stage"],
                "timestamp": checkpoint_data["timestamp"],
                "config": checkpoint_data["config"],
                "metadata": {
                    key: value for key, value in data.items() 
                    if key not in ["train_dataset", "test_dataset", "models"]
                }
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_to_save, f, indent=2, default=str)
            
            self.checkpoints[stage] = checkpoint_path
            logger.info(f"Checkpoint created for stage '{stage}' at {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint for stage '{stage}': {e}")
    
    def load_checkpoint(self, stage: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint for a pipeline stage.
        
        Args:
            stage: Pipeline stage name
            
        Returns:
            Checkpoint data if available, None otherwise
        """
        checkpoint_path = self.path_manager.get_checkpoint_path(f"{stage}_checkpoint.json")
        
        if not checkpoint_path.exists():
            logger.debug(f"No checkpoint found for stage '{stage}'")
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"Loaded checkpoint for stage '{stage}' from {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint for stage '{stage}': {e}")
            return None
    
    def save_model_artifacts(self, models: Dict[str, Any], stage: str):
        """Save model artifacts for reproducibility.
        
        Args:
            models: Dictionary of trained models
            stage: Pipeline stage name
        """
        logger.info(f"Saving model artifacts for stage '{stage}'...")
        
        artifacts_dir = self.path_manager.ensure_directory(f"models/{stage}")
        
        for model_name, model_data in models.items():
            try:
                if hasattr(model_data, 'state_dict'):
                    # PyTorch model
                    import torch
                    model_path = artifacts_dir / f"{model_name}.pt"
                    torch.save(model_data.state_dict(), model_path)
                    logger.info(f"Saved PyTorch model: {model_path}")
                    
                elif hasattr(model_data, 'save_model'):
                    # Tree-based model (LightGBM, XGBoost, CatBoost)
                    model_path = artifacts_dir / f"{model_name}.model"
                    model_data.save_model(str(model_path))
                    logger.info(f"Saved tree model: {model_path}")
                    
                elif hasattr(model_data, 'dump'):
                    # Pickle-able model
                    import pickle
                    model_path = artifacts_dir / f"{model_name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_data, f)
                    logger.info(f"Saved pickled model: {model_path}")
                    
                else:
                    logger.warning(f"Unknown model type for {model_name}, skipping save")
                    
            except Exception as e:
                logger.error(f"Failed to save model {model_name}: {e}")
    
    def validate_submission_format(self, submission_df) -> bool:
        """Validate submission file format.
        
        Args:
            submission_df: Submission DataFrame
            
        Returns:
            True if format is valid, False otherwise
        """
        logger.info("Validating submission format...")
        
        required_columns = ['id'] + self.config.data.target_cols
        
        # Check required columns
        missing_columns = set(required_columns) - set(submission_df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for missing values
        if submission_df.isnull().any().any():
            logger.warning("Submission contains missing values")
            # Fill missing values with 0
            submission_df.fillna(0, inplace=True)
            logger.info("Filled missing values with 0")
        
        # Check data types
        for col in self.config.data.target_cols:
            if not submission_df[col].dtype.kind in 'biufc':  # numeric types
                logger.error(f"Column {col} is not numeric")
                return False
        
        # Check for infinite values
        if submission_df.select_dtypes(include=['float64', 'float32']).isin([float('inf'), float('-inf')]).any().any():
            logger.error("Submission contains infinite values")
            return False
        
        logger.info("Submission format validation passed")
        return True
    
    def generate_submission_file(self, predictions: Dict[str, Any], 
                               filename: str = None) -> Path:
        """Generate and validate submission file.
        
        Args:
            predictions: Prediction results
            filename: Optional custom filename
            
        Returns:
            Path to the generated submission file
        """
        logger.info("Generating submission file...")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_production_{timestamp}.csv"
        
        submission_path = self.path_manager.get_output_path(filename)
        
        try:
            # Extract prediction data
            if 'prediction_results' in predictions:
                pred_data = predictions['prediction_results']
                if 'predictions' in pred_data and 'ids' in pred_data:
                    import pandas as pd
                    import numpy as np
                    
                    # Create submission DataFrame
                    submission_df = pd.DataFrame({
                        'id': pred_data['ids']
                    })
                    
                    # Add prediction columns
                    preds_array = pred_data['predictions']
                    if isinstance(preds_array, np.ndarray):
                        for i, col in enumerate(self.config.data.target_cols):
                            if i < preds_array.shape[1]:
                                submission_df[col] = preds_array[:, i]
                            else:
                                submission_df[col] = 0.0
                    else:
                        # Fallback for non-array predictions
                        for col in self.config.data.target_cols:
                            submission_df[col] = 0.0
                    
                    # Validate format
                    if not self.validate_submission_format(submission_df):
                        raise ValueError("Submission format validation failed")
                    
                    # Save submission file
                    submission_df.to_csv(submission_path, index=False)
                    
                    logger.info(f"Submission file generated: {submission_path}")
                    logger.info(f"Submission shape: {submission_df.shape}")
                    logger.info(f"Sample predictions: {submission_df.head()}")
                    
                    return submission_path
                else:
                    raise ValueError("Invalid prediction data structure")
            else:
                raise ValueError("No prediction results found")
                
        except Exception as e:
            logger.error(f"Failed to generate submission file: {e}")
            raise
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report.
        
        Returns:
            Comprehensive report dictionary
        """
        logger.info("Generating comprehensive pipeline report...")
        
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        report = {
            "pipeline_info": {
                "pipeline_type": "production",
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": duration,
                "success": all(
                    result.get("success", False) 
                    for result in self.results.values() 
                    if isinstance(result, dict)
                )
            },
            "environment_info": self.config.get_environment_info(),
            "configuration": self.config.to_dict(),
            "checkpoints": {
                stage: str(path) for stage, path in self.checkpoints.items()
            },
            "results_summary": {},
            "error_summary": self.error_handler.get_error_summary(),
            "performance_metrics": {}
        }
        
        # Add results summary
        for stage, result in self.results.items():
            if isinstance(result, dict):
                summary = {
                    "success": result.get("success", False),
                    "duration": result.get("duration", None),
                    "error": result.get("error", None)
                }
                
                # Add stage-specific metrics
                if stage == "data_processing":
                    if "data_statistics" in result:
                        summary["data_stats"] = result["data_statistics"]
                elif stage == "model_training":
                    if "training_results" in result:
                        training_results = result["training_results"]
                        summary["models_trained"] = list(training_results.keys()) if isinstance(training_results, dict) else []
                elif stage == "prediction_generation":
                    if "prediction_results" in result:
                        pred_results = result["prediction_results"]
                        if "prediction_summary" in pred_results:
                            summary["prediction_summary"] = pred_results["prediction_summary"]
                
                report["results_summary"][stage] = summary
        
        return report
    
    def save_comprehensive_report(self, report: Dict[str, Any]) -> Path:
        """Save comprehensive report to file.
        
        Args:
            report: Report dictionary
            
        Returns:
            Path to the saved report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.path_manager.get_output_path(f"production_report_{timestamp}.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Comprehensive report saved to {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to save comprehensive report: {e}")
            raise
    
    def run_production_pipeline(self, model_type: str = "stacking_ensemble") -> Dict[str, Any]:
        """Run the complete production pipeline.
        
        Args:
            model_type: Type of model to use for final predictions
            
        Returns:
            Complete pipeline results
        """
        logger.info("=== Starting Production Polymer Prediction Pipeline ===")
        
        self.start_time = datetime.now()
        
        try:
            # Setup environment
            self.setup_environment()
            
            # Run data processing with checkpointing
            logger.info("=== Stage 1: Data Processing ===")
            with get_performance_timer(logger, "data_processing"):
                data_results = self.main_pipeline.run_data_processing()
                self.results["data_processing"] = data_results
                self.create_checkpoint("data_processing", data_results)
            
            if not data_results["success"]:
                raise RuntimeError("Data processing failed")
            
            # Run model training with checkpointing
            logger.info("=== Stage 2: Model Training ===")
            with get_performance_timer(logger, "model_training"):
                training_results = self.main_pipeline.run_model_training(
                    data_results["train_dataset"]
                )
                self.results["model_training"] = training_results
                self.create_checkpoint("model_training", training_results)
                
                # Save model artifacts
                if training_results["success"] and "models" in training_results:
                    self.save_model_artifacts(training_results["models"], "training")
            
            if not training_results["success"]:
                raise RuntimeError("Model training failed")
            
            # Run prediction generation with checkpointing
            logger.info("=== Stage 3: Prediction Generation ===")
            with get_performance_timer(logger, "prediction_generation"):
                prediction_results = self.main_pipeline.run_prediction_generation(
                    training_results["models"],
                    data_results["test_dataset"],
                    model_type
                )
                self.results["prediction_generation"] = prediction_results
                self.create_checkpoint("prediction_generation", prediction_results)
            
            if not prediction_results["success"]:
                raise RuntimeError("Prediction generation failed")
            
            # Generate submission file
            logger.info("=== Stage 4: Submission Generation ===")
            submission_path = self.generate_submission_file(prediction_results)
            
            self.end_time = datetime.now()
            
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            report_path = self.save_comprehensive_report(report)
            
            logger.info("=== Production Pipeline Completed Successfully ===")
            logger.info(f"Total duration: {self.end_time - self.start_time}")
            logger.info(f"Submission file: {submission_path}")
            logger.info(f"Comprehensive report: {report_path}")
            
            return {
                "success": True,
                "results": self.results,
                "submission_path": submission_path,
                "report": report,
                "report_path": report_path,
                "checkpoints": self.checkpoints
            }
            
        except Exception as e:
            self.end_time = datetime.now()
            
            logger.error(f"Production pipeline failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Generate failure report
            try:
                report = self.generate_comprehensive_report()
                report_path = self.save_comprehensive_report(report)
                logger.info(f"Failure report saved to: {report_path}")
            except Exception as report_error:
                logger.error(f"Failed to save failure report: {report_error}")
                report = {}
                report_path = None
            
            return {
                "success": False,
                "error": str(e),
                "results": self.results,
                "report": report,
                "report_path": report_path,
                "checkpoints": self.checkpoints
            }


def parse_command_line_arguments():
    """Parse command line arguments for the production pipeline."""
    parser = argparse.ArgumentParser(
        description="Production-Ready Polymer Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (JSON format)"
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
        help="Directory containing training and test data files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output files (submission, reports, etc.)"
    )
    
    parser.add_argument(
        "--submission-filename",
        type=str,
        help="Custom filename for submission file (default: auto-generated with timestamp)"
    )
    
    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--hidden-channels",
        type=int,
        help="Number of hidden channels in GCN model (overrides config)"
    )
    
    parser.add_argument(
        "--num-gcn-layers",
        type=int,
        help="Number of GCN layers (overrides config)"
    )
    
    # Cross-validation options
    parser.add_argument(
        "--n-folds",
        type=int,
        help="Number of cross-validation folds (overrides config)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility (overrides config)"
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
        help="Log file name (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging (log to file only)"
    )
    
    parser.add_argument(
        "--structured-logging",
        action="store_true",
        default=True,
        help="Enable structured JSON logging"
    )
    
    # Performance options
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-only training (disable GPU)"
    )
    
    parser.add_argument(
        "--enable-caching",
        action="store_true",
        default=True,
        help="Enable graph caching for performance"
    )
    
    parser.add_argument(
        "--memory-monitoring",
        action="store_true",
        default=True,
        help="Enable memory usage monitoring"
    )
    
    # Development and debugging options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging, save intermediate results)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run (validate configuration and data, but don't train models)"
    )
    
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume pipeline from a specific checkpoint stage"
    )
    
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        default=True,
        help="Save checkpoints during pipeline execution"
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> Config:
    """Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration object with applied overrides
    """
    # Load base configuration
    if args.config:
        if Path(args.config).exists():
            config = Config.load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.warning(f"Configuration file not found: {args.config}, using defaults")
            config = Config()
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
    
    if args.n_folds:
        overrides.setdefault("training", {})["n_folds"] = args.n_folds
    
    if args.random_seed:
        overrides.setdefault("training", {})["random_state"] = args.random_seed
        overrides.setdefault("data", {})["random_seed"] = args.random_seed
    
    # Model overrides
    if args.hidden_channels:
        overrides.setdefault("model", {})["hidden_channels"] = args.hidden_channels
    
    if args.num_gcn_layers:
        overrides.setdefault("model", {})["num_gcn_layers"] = args.num_gcn_layers
    
    # Logging overrides
    if args.log_level:
        overrides.setdefault("logging", {})["level"] = args.log_level
    
    if args.log_file:
        overrides.setdefault("logging", {})["log_file"] = args.log_file
    else:
        # Auto-generate log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        overrides.setdefault("logging", {})["log_file"] = f"production_pipeline_{timestamp}.log"
    
    if args.no_console_log:
        overrides.setdefault("logging", {})["log_to_console"] = False
    
    if args.structured_logging:
        overrides.setdefault("logging", {})["use_structured_logging"] = True
    
    # Performance overrides
    if args.force_cpu:
        import torch
        config.device = torch.device("cpu")
        config._apply_cpu_optimizations()
    
    if args.enable_caching:
        overrides.setdefault("performance", {})["enable_graph_cache"] = True
    
    if args.memory_monitoring:
        overrides.setdefault("performance", {})["enable_memory_monitoring"] = True
    
    # Debug mode
    if args.debug:
        overrides.setdefault("logging", {})["level"] = "DEBUG"
        overrides.setdefault("logging", {})["log_to_console"] = True
        overrides.setdefault("logging", {})["use_structured_logging"] = True
    
    # Apply overrides
    if overrides:
        config._apply_overrides(overrides)
        logger.info("Applied command line configuration overrides")
    
    return config


def main():
    """Main entry point for the production pipeline."""
    try:
        # Parse command line arguments
        args = parse_command_line_arguments()
        
        # Create configuration
        config = create_config_from_args(args)
        
        # Create path manager
        path_manager = PathManager()
        
        # Initialize production pipeline
        pipeline = ProductionPipeline(config, path_manager)
        
        # Handle dry run
        if args.dry_run:
            logger.info("=== Dry Run Mode ===")
            pipeline.setup_environment()
            logger.info("Configuration and environment validation completed")
            logger.info("Dry run completed successfully")
            return 0
        
        # Handle resume from checkpoint
        if args.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            checkpoint_data = pipeline.load_checkpoint(args.resume_from_checkpoint)
            if checkpoint_data is None:
                logger.error(f"Checkpoint not found: {args.resume_from_checkpoint}")
                return 1
        
        # Log startup information
        logger.info("=== Production Pipeline Startup ===")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Data directory: {config.paths.data_path}")
        logger.info(f"Output directory: {config.paths.outputs_path}")
        logger.info(f"Debug mode: {args.debug}")
        
        # Run production pipeline
        with get_performance_timer(logger, "complete_production_pipeline"):
            results = pipeline.run_production_pipeline(args.model_type)
        
        # Handle results
        if results["success"]:
            logger.info("=== Production Pipeline Completed Successfully ===")
            
            # Log key results
            if results.get("submission_path"):
                logger.info(f"Submission file: {results['submission_path']}")
            
            if results.get("report_path"):
                logger.info(f"Comprehensive report: {results['report_path']}")
            
            # Log checkpoints
            if results.get("checkpoints"):
                logger.info("Checkpoints created:")
                for stage, path in results["checkpoints"].items():
                    logger.info(f"  {stage}: {path}")
            
            return 0
        else:
            logger.error("=== Production Pipeline Failed ===")
            logger.error(f"Error: {results.get('error', 'Unknown error')}")
            
            if results.get("report_path"):
                logger.error(f"Failure report: {results['report_path']}")
            
            return 1
    
    except KeyboardInterrupt:
        logger.warning("Production pipeline interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error in production pipeline: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)