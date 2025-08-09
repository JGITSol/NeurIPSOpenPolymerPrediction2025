"""Prediction pipeline for polymer prediction models."""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from polymer_prediction.config.config import Config
from polymer_prediction.data.dataset import PolymerDataset
from polymer_prediction.training.trainer import predict
from polymer_prediction.utils.logging import get_logger
from polymer_prediction.utils.path_manager import PathManager
from polymer_prediction.utils.error_handling import ErrorHandler, DeviceManager

logger = get_logger(__name__)


class PredictionPipeline:
    """Prediction pipeline for polymer prediction models."""
    
    def __init__(self, config: Config, path_manager: Optional[PathManager] = None):
        """Initialize prediction pipeline.
        
        Args:
            config: Configuration object
            path_manager: Path manager for file operations
        """
        self.config = config
        self.path_manager = path_manager or PathManager()
        self.error_handler = ErrorHandler()
        self.device_manager = DeviceManager(self.error_handler)
        
        logger.info("PredictionPipeline initialized")
    
    def create_test_loader(self, test_dataset: PolymerDataset) -> DataLoader:
        """Create data loader for test data.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Test data loader
        """
        logger.info("Creating test data loader...")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory and self.config.device.type == 'cuda',
            drop_last=False
        )
        
        logger.info(f"Test data loader created with {len(test_dataset)} samples")
        
        return test_loader
    
    def predict_gcn(self, model: torch.nn.Module, test_loader: DataLoader) -> Tuple[List[int], np.ndarray]:
        """Generate predictions using GCN model.
        
        Args:
            model: Trained GCN model
            test_loader: Test data loader
            
        Returns:
            Tuple of (test_ids, predictions)
        """
        logger.info("Generating GCN predictions...")
        
        model.eval()
        test_ids, predictions = predict(model, test_loader, self.config.device)
        
        logger.info(f"GCN predictions generated: {len(test_ids)} samples, {predictions.shape[1]} targets")
        
        return test_ids, predictions
    
    def predict_tree_ensemble(self, model: Any, test_dataset: PolymerDataset) -> Tuple[List[int], np.ndarray]:
        """Generate predictions using tree ensemble model.
        
        Args:
            model: Trained tree ensemble model
            test_dataset: Test dataset
            
        Returns:
            Tuple of (test_ids, predictions)
        """
        logger.info("Generating tree ensemble predictions...")
        
        # Extract features from test dataset
        features = []
        test_ids = []
        
        for data in test_dataset:
            if data is not None:
                # Use molecular descriptors as features for tree models
                feature_vector = data.x.mean(dim=0).numpy()  # Simple aggregation
                features.append(feature_vector)
                
                # Extract ID
                if hasattr(data, 'id'):
                    test_ids.append(int(data.id))
                else:
                    test_ids.append(len(test_ids))
        
        if not features:
            raise ValueError("No valid features extracted for tree ensemble prediction")
        
        features = np.array(features)
        
        # Generate predictions
        predictions = model.predict(features)
        
        logger.info(f"Tree ensemble predictions generated: {len(test_ids)} samples, {predictions.shape[1]} targets")
        
        return test_ids, predictions
    
    def predict_stacking_ensemble(self, model: Any, test_dataset: PolymerDataset) -> Tuple[List[int], np.ndarray]:
        """Generate predictions using stacking ensemble model.
        
        Args:
            model: Trained stacking ensemble model
            test_dataset: Test dataset
            
        Returns:
            Tuple of (test_ids, predictions)
        """
        logger.info("Generating stacking ensemble predictions...")
        
        # Create test loader for stacking ensemble
        test_loader = self.create_test_loader(test_dataset)
        
        # Generate predictions
        test_ids, predictions = model.predict(test_loader)
        
        logger.info(f"Stacking ensemble predictions generated: {len(test_ids)} samples, {predictions.shape[1]} targets")
        
        return test_ids, predictions
    
    def create_submission_dataframe(self, test_ids: List[int], predictions: np.ndarray) -> pd.DataFrame:
        """Create submission DataFrame from predictions.
        
        Args:
            test_ids: List of test sample IDs
            predictions: Prediction array
            
        Returns:
            Submission DataFrame
        """
        logger.info("Creating submission DataFrame...")
        
        # Create base DataFrame with IDs
        submission = pd.DataFrame({self.config.data.id_column: test_ids})
        
        # Add predictions for each target column
        for i, col in enumerate(self.config.data.target_cols):
            if i < predictions.shape[1]:
                submission[col] = predictions[:, i]
            else:
                # Fallback value if prediction is missing
                submission[col] = 0.0
                logger.warning(f"Missing prediction for target {col}, using fallback value 0.0")
        
        logger.info(f"Submission DataFrame created: {submission.shape}")
        
        return submission
    
    def validate_submission(self, submission: pd.DataFrame) -> pd.DataFrame:
        """Validate submission DataFrame format.
        
        Args:
            submission: Submission DataFrame
            
        Returns:
            Validated submission DataFrame
        """
        logger.info("Validating submission format...")
        
        # Check required columns
        required_cols = [self.config.data.id_column] + self.config.data.target_cols
        missing_cols = [col for col in required_cols if col not in submission.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in submission: {missing_cols}")
        
        # Check for missing values
        if submission.isnull().any().any():
            logger.warning("Submission contains missing values, filling with 0.0")
            submission = submission.fillna(0.0)
        
        # Check for infinite values
        numeric_cols = self.config.data.target_cols
        for col in numeric_cols:
            if col in submission.columns:
                if np.isinf(submission[col]).any():
                    logger.warning(f"Column {col} contains infinite values, replacing with 0.0")
                    submission[col] = submission[col].replace([np.inf, -np.inf], 0.0)
        
        # Ensure proper data types
        submission[self.config.data.id_column] = submission[self.config.data.id_column].astype(int)
        for col in numeric_cols:
            if col in submission.columns:
                submission[col] = submission[col].astype(float)
        
        logger.info("Submission validation completed")
        
        return submission
    
    def save_submission(self, submission: pd.DataFrame, filename: str) -> Path:
        """Save submission DataFrame to CSV file.
        
        Args:
            submission: Submission DataFrame
            filename: Output filename
            
        Returns:
            Path to the saved submission file
        """
        output_path = self.path_manager.get_output_path(filename)
        
        try:
            submission.to_csv(output_path, index=False)
            logger.info(f"Submission saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving submission: {e}")
            raise
        
        return output_path
    
    def generate_prediction_summary(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Generate summary statistics for predictions.
        
        Args:
            predictions: Prediction array
            
        Returns:
            Dictionary containing prediction statistics
        """
        summary = {
            "num_samples": predictions.shape[0],
            "num_targets": predictions.shape[1],
            "target_statistics": {}
        }
        
        for i, col in enumerate(self.config.data.target_cols):
            if i < predictions.shape[1]:
                col_preds = predictions[:, i]
                summary["target_statistics"][col] = {
                    "mean": float(np.mean(col_preds)),
                    "std": float(np.std(col_preds)),
                    "min": float(np.min(col_preds)),
                    "max": float(np.max(col_preds)),
                    "median": float(np.median(col_preds))
                }
        
        return summary
    
    def run_prediction_pipeline(self, models: Dict[str, Any], test_dataset: PolymerDataset, 
                               model_type: str = "stacking_ensemble") -> Dict[str, Any]:
        """Run the complete prediction pipeline.
        
        Args:
            models: Dictionary of trained models
            test_dataset: Test dataset
            model_type: Type of model to use for predictions
            
        Returns:
            Prediction results dictionary
        """
        logger.info(f"Running prediction pipeline with {model_type} model...")
        
        results = {}
        
        try:
            # Generate predictions based on model type
            if model_type == "gcn" and "gcn" in models:
                test_loader = self.create_test_loader(test_dataset)
                test_ids, predictions = self.predict_gcn(models["gcn"], test_loader)
                
            elif model_type == "tree_ensemble" and "tree_ensemble" in models:
                test_ids, predictions = self.predict_tree_ensemble(models["tree_ensemble"], test_dataset)
                
            elif model_type == "stacking_ensemble" and "stacking_ensemble" in models:
                test_ids, predictions = self.predict_stacking_ensemble(models["stacking_ensemble"], test_dataset)
                
            else:
                # Fallback to available model
                available_models = list(models.keys())
                if not available_models:
                    raise ValueError("No trained models available for prediction")
                
                fallback_model = available_models[0]
                logger.warning(f"Requested model {model_type} not available, using {fallback_model}")
                
                if fallback_model == "gcn":
                    test_loader = self.create_test_loader(test_dataset)
                    test_ids, predictions = self.predict_gcn(models[fallback_model], test_loader)
                else:
                    test_ids, predictions = self.predict_tree_ensemble(models[fallback_model], test_dataset)
            
            # Create submission DataFrame
            submission = self.create_submission_dataframe(test_ids, predictions)
            
            # Validate submission
            submission = self.validate_submission(submission)
            
            # Generate prediction summary
            pred_summary = self.generate_prediction_summary(predictions)
            
            # Save submission
            submission_filename = f"submission_{model_type}.csv"
            submission_path = self.save_submission(submission, submission_filename)
            
            results = {
                "test_ids": test_ids,
                "predictions": predictions,
                "submission": submission,
                "submission_path": submission_path,
                "prediction_summary": pred_summary,
                "model_type": model_type
            }
            
            logger.info("Prediction pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}")
            raise
        
        return results