"""Main pipeline orchestrating the complete polymer prediction workflow."""

import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from polymer_prediction.config.config import Config
from polymer_prediction.pipeline.data_pipeline import DataPipeline
from polymer_prediction.pipeline.training_pipeline import TrainingPipeline
from polymer_prediction.pipeline.prediction_pipeline import PredictionPipeline
from polymer_prediction.utils.logging import get_logger, setup_logging
from polymer_prediction.utils.path_manager import PathManager
from polymer_prediction.utils.error_handling import ErrorHandler

logger = get_logger(__name__)


class MainPipeline:
    """Main pipeline orchestrating the complete polymer prediction workflow."""
    
    def __init__(self, config: Optional[Config] = None, path_manager: Optional[PathManager] = None):
        """Initialize main pipeline.
        
        Args:
            config: Configuration object (optional, will create default if None)
            path_manager: Path manager for file operations (optional)
        """
        self.config = config or Config()
        self.path_manager = path_manager or PathManager()
        self.error_handler = ErrorHandler()
        
        # Initialize sub-pipelines
        self.data_pipeline = DataPipeline(self.config, self.path_manager)
        self.training_pipeline = TrainingPipeline(self.config, self.path_manager)
        self.prediction_pipeline = PredictionPipeline(self.config, self.path_manager)
        
        # Pipeline state
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        logger.info("MainPipeline initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = None
        if self.config.logging.log_to_file and self.config.logging.log_file:
            log_file = str(self.path_manager.get_log_path(self.config.logging.log_file))
        
        setup_logging(
            log_level=self.config.logging.level,
            log_file=log_file
        )
        
        logger.info("Logging setup completed")
    
    def log_environment_info(self):
        """Log environment and configuration information."""
        logger.info("=== Environment Information ===")
        
        env_info = self.config.get_environment_info()
        for key, value in env_info.items():
            logger.info(f"{key}: {value}")
        
        logger.info("=== Configuration ===")
        config_dict = self.config.to_dict()
        for section, values in config_dict.items():
            logger.info(f"{section}: {values}")
    
    def save_pipeline_config(self):
        """Save pipeline configuration to file."""
        config_path = self.path_manager.get_output_path("pipeline_config.json")
        self.config.save_config(str(config_path))
        logger.info(f"Pipeline configuration saved to {config_path}")
    
    def run_data_processing(self) -> Dict[str, Any]:
        """Run data processing pipeline.
        
        Returns:
            Data processing results
        """
        logger.info("=== Starting Data Processing ===")
        
        try:
            train_dataset, test_dataset, data_stats = self.data_pipeline.run_pipeline()
            
            results = {
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "data_statistics": data_stats,
                "success": True
            }
            
            logger.info("Data processing completed successfully")
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            results = {
                "success": False,
                "error": str(e)
            }
            raise
        
        return results
    
    def run_model_training(self, train_dataset, val_dataset=None) -> Dict[str, Any]:
        """Run model training pipeline.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            
        Returns:
            Training results
        """
        logger.info("=== Starting Model Training ===")
        
        try:
            training_results = self.training_pipeline.run_training_pipeline(
                train_dataset, val_dataset
            )
            
            results = {
                "training_results": training_results,
                "models": self.training_pipeline.models,
                "training_history": self.training_pipeline.training_history,
                "success": True
            }
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            results = {
                "success": False,
                "error": str(e)
            }
            raise
        
        return results
    
    def run_prediction_generation(self, models: Dict[str, Any], test_dataset, 
                                 model_type: str = "stacking_ensemble") -> Dict[str, Any]:
        """Run prediction generation pipeline.
        
        Args:
            models: Dictionary of trained models
            test_dataset: Test dataset
            model_type: Type of model to use for predictions
            
        Returns:
            Prediction results
        """
        logger.info("=== Starting Prediction Generation ===")
        
        try:
            prediction_results = self.prediction_pipeline.run_prediction_pipeline(
                models, test_dataset, model_type
            )
            
            results = {
                "prediction_results": prediction_results,
                "success": True
            }
            
            logger.info("Prediction generation completed successfully")
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            results = {
                "success": False,
                "error": str(e)
            }
            raise
        
        return results
    
    def generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report.
        
        Returns:
            Pipeline report dictionary
        """
        duration = None
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
        
        report = {
            "pipeline_info": {
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
            "results": {}
        }
        
        # Add results from each pipeline stage
        for stage, result in self.results.items():
            if isinstance(result, dict):
                # Remove large objects for report
                filtered_result = {}
                for key, value in result.items():
                    if key not in ["train_dataset", "test_dataset", "models"]:
                        filtered_result[key] = value
                report["results"][stage] = filtered_result
        
        # Add error summary
        error_summary = self.error_handler.get_error_summary()
        report["error_summary"] = error_summary
        
        return report
    
    def save_pipeline_report(self, report: Dict[str, Any]) -> Path:
        """Save pipeline report to file.
        
        Args:
            report: Pipeline report dictionary
            
        Returns:
            Path to the saved report file
        """
        import json
        
        report_path = self.path_manager.get_output_path("pipeline_report.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Pipeline report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving pipeline report: {e}")
            raise
        
        return report_path
    
    def run_complete_pipeline(self, model_type: str = "stacking_ensemble") -> Dict[str, Any]:
        """Run the complete polymer prediction pipeline.
        
        Args:
            model_type: Type of model to use for final predictions
            
        Returns:
            Complete pipeline results
        """
        logger.info("=== Starting Complete Polymer Prediction Pipeline ===")
        
        self.start_time = datetime.now()
        
        try:
            # Setup logging
            self.setup_logging()
            
            # Log environment information
            self.log_environment_info()
            
            # Save configuration
            self.save_pipeline_config()
            
            # Run data processing
            data_results = self.run_data_processing()
            self.results["data_processing"] = data_results
            
            if not data_results["success"]:
                raise RuntimeError("Data processing failed")
            
            # Run model training
            training_results = self.run_model_training(
                data_results["train_dataset"]
            )
            self.results["model_training"] = training_results
            
            if not training_results["success"]:
                raise RuntimeError("Model training failed")
            
            # Run prediction generation
            prediction_results = self.run_prediction_generation(
                training_results["models"],
                data_results["test_dataset"],
                model_type
            )
            self.results["prediction_generation"] = prediction_results
            
            if not prediction_results["success"]:
                raise RuntimeError("Prediction generation failed")
            
            self.end_time = datetime.now()
            
            # Generate and save pipeline report
            report = self.generate_pipeline_report()
            report_path = self.save_pipeline_report(report)
            
            logger.info("=== Complete Pipeline Finished Successfully ===")
            logger.info(f"Total duration: {self.end_time - self.start_time}")
            logger.info(f"Pipeline report saved to: {report_path}")
            
            return {
                "success": True,
                "results": self.results,
                "report": report,
                "report_path": report_path
            }
            
        except Exception as e:
            self.end_time = datetime.now()
            
            logger.error(f"Pipeline failed: {e}")
            
            # Generate report even for failed pipeline
            try:
                report = self.generate_pipeline_report()
                report_path = self.save_pipeline_report(report)
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
                "report_path": report_path
            }


def run_polymer_prediction_pipeline(config_path: Optional[str] = None, 
                                   model_type: str = "stacking_ensemble") -> Dict[str, Any]:
    """Convenience function to run the complete pipeline.
    
    Args:
        config_path: Path to configuration file (optional)
        model_type: Type of model to use for predictions
        
    Returns:
        Pipeline results
    """
    # Load configuration
    if config_path:
        config = Config.load_config(config_path)
    else:
        config = Config()
    
    # Create and run pipeline
    pipeline = MainPipeline(config)
    results = pipeline.run_complete_pipeline(model_type)
    
    return results