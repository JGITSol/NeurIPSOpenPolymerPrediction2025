"""Data processing pipeline for polymer prediction."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

from polymer_prediction.config.config import Config
from polymer_prediction.data.dataset import PolymerDataset
from polymer_prediction.data.validation import validate_dataframe
from polymer_prediction.utils.logging import get_logger
from polymer_prediction.utils.path_manager import PathManager
from polymer_prediction.utils.error_handling import ErrorHandler, SMILESValidator

logger = get_logger(__name__)


class DataPipeline:
    """Data processing pipeline for polymer prediction."""
    
    def __init__(self, config: Config, path_manager: Optional[PathManager] = None):
        """Initialize data pipeline.
        
        Args:
            config: Configuration object
            path_manager: Path manager for file operations
        """
        self.config = config
        self.path_manager = path_manager or PathManager()
        self.error_handler = ErrorHandler()
        self.smiles_validator = SMILESValidator(self.error_handler)
        
        logger.info("DataPipeline initialized")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data.
        
        Returns:
            Tuple of (train_df, test_df)
            
        Raises:
            FileNotFoundError: If data files are not found
            ValueError: If data validation fails
        """
        logger.info("Loading training and test data...")
        
        # Get data file paths
        train_path = self.path_manager.get_data_path(self.config.paths.train_file)
        test_path = self.path_manager.get_data_path(self.config.paths.test_file)
        
        # Check if files exist
        if not self.path_manager.file_exists(train_path):
            raise FileNotFoundError(f"Training data file not found: {train_path}")
        
        if not self.path_manager.file_exists(test_path):
            raise FileNotFoundError(f"Test data file not found: {test_path}")
        
        # Load data
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logger.info(f"Loaded training data: {train_df.shape}")
            logger.info(f"Loaded test data: {test_df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            raise
        
        return train_df, test_df
    
    def validate_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validate and clean data.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of validated (train_df, test_df)
        """
        logger.info("Validating data...")
        
        # Define required columns
        required_train_cols = [self.config.data.id_column, self.config.data.smiles_column] + self.config.data.target_cols
        required_test_cols = [self.config.data.id_column, self.config.data.smiles_column]
        
        # Validate DataFrame structure
        from polymer_prediction.data.validation import DataValidationConfig
        
        # Create validation config for training data
        train_config = DataValidationConfig(
            required_columns=required_train_cols,
            smiles_column=self.config.data.smiles_column,
            target_column=self.config.data.target_cols[0] if self.config.data.target_cols else None,
            min_samples=5
        )
        
        # Create validation config for test data
        test_config = DataValidationConfig(
            required_columns=required_test_cols,
            smiles_column=self.config.data.smiles_column,
            min_samples=1
        )
        
        # Validate DataFrames
        train_report = validate_dataframe(train_df, train_config)
        test_report = validate_dataframe(test_df, test_config)
        
        # Log validation results
        if not train_report.is_valid:
            logger.warning("Training data validation issues found:")
            for error in train_report.errors:
                logger.error(f"  {error}")
            for warning in train_report.warnings:
                logger.warning(f"  {warning}")
        
        if not test_report.is_valid:
            logger.warning("Test data validation issues found:")
            for error in test_report.errors:
                logger.error(f"  {error}")
            for warning in test_report.warnings:
                logger.warning(f"  {warning}")
        
        # Continue with processing even if validation has warnings
        logger.info(f"Training data: {train_report.valid_samples}/{train_report.total_samples} valid samples")
        logger.info(f"Test data: {test_report.valid_samples}/{test_report.total_samples} valid samples")
        
        # Validate SMILES strings
        logger.info("Validating SMILES strings...")
        train_df = self.smiles_validator.filter_valid_smiles(train_df, self.config.data.smiles_column)
        test_df = self.smiles_validator.filter_valid_smiles(test_df, self.config.data.smiles_column)
        
        # Check for empty DataFrames
        if len(train_df) == 0:
            raise ValueError("No valid training samples after SMILES validation")
        
        if len(test_df) == 0:
            raise ValueError("No valid test samples after SMILES validation")
        
        logger.info(f"Data validation completed: Train {train_df.shape}, Test {test_df.shape}")
        
        return train_df, test_df
    
    def create_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[PolymerDataset, PolymerDataset]:
        """Create PyTorch datasets from DataFrames.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        logger.info("Creating PyTorch datasets...")
        
        try:
            train_dataset = PolymerDataset(
                train_df, 
                target_cols=self.config.data.target_cols, 
                is_test=False
            )
            
            test_dataset = PolymerDataset(
                test_df, 
                target_cols=self.config.data.target_cols, 
                is_test=True
            )
            
            logger.info(f"Created datasets: Train {len(train_dataset)}, Test {len(test_dataset)}")
            
            return train_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error creating datasets: {e}")
            raise
    
    def get_data_statistics(self, train_df: pd.DataFrame) -> dict:
        """Get statistics about the training data.
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            Dictionary containing data statistics
        """
        stats = {
            "num_samples": len(train_df),
            "num_features": len(self.config.data.target_cols),
            "target_statistics": {}
        }
        
        # Calculate statistics for each target column
        for col in self.config.data.target_cols:
            if col in train_df.columns:
                col_data = train_df[col].dropna()
                stats["target_statistics"][col] = {
                    "count": len(col_data),
                    "missing": len(train_df) - len(col_data),
                    "mean": float(col_data.mean()) if len(col_data) > 0 else 0.0,
                    "std": float(col_data.std()) if len(col_data) > 0 else 0.0,
                    "min": float(col_data.min()) if len(col_data) > 0 else 0.0,
                    "max": float(col_data.max()) if len(col_data) > 0 else 0.0
                }
        
        logger.info(f"Data statistics calculated: {stats['num_samples']} samples, {stats['num_features']} targets")
        
        return stats
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           output_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Save processed data to files.
        
        Args:
            train_df: Processed training DataFrame
            test_df: Processed test DataFrame
            output_dir: Output directory (optional)
            
        Returns:
            Tuple of (train_file_path, test_file_path)
        """
        if output_dir:
            output_path = self.path_manager.ensure_directory(output_dir)
        else:
            output_path = self.path_manager.ensure_directory("processed_data")
        
        train_file_path = output_path / "processed_train.csv"
        test_file_path = output_path / "processed_test.csv"
        
        train_df.to_csv(train_file_path, index=False)
        test_df.to_csv(test_file_path, index=False)
        
        logger.info(f"Processed data saved to {output_path}")
        
        return train_file_path, test_file_path
    
    def run_pipeline(self) -> Tuple[PolymerDataset, PolymerDataset, dict]:
        """Run the complete data processing pipeline.
        
        Returns:
            Tuple of (train_dataset, test_dataset, data_statistics)
        """
        logger.info("Running data processing pipeline...")
        
        # Load data
        train_df, test_df = self.load_data()
        
        # Validate data
        train_df, test_df = self.validate_data(train_df, test_df)
        
        # Get statistics
        stats = self.get_data_statistics(train_df)
        
        # Create datasets
        train_dataset, test_dataset = self.create_datasets(train_df, test_df)
        
        logger.info("Data processing pipeline completed successfully")
        
        return train_dataset, test_dataset, stats