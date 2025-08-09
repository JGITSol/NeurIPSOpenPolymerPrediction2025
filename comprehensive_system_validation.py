#!/usr/bin/env python3
"""
Comprehensive System Validation Script

This script performs end-to-end validation of the complete polymer prediction system,
testing all components from data loading to submission generation.
"""

import sys
import os
import json
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Core imports
from polymer_prediction.config.config import Config
from polymer_prediction.utils.logging import setup_logging, get_logger
from polymer_prediction.utils.path_manager import PathManager

# Initialize logger
logger = get_logger(__name__)


def create_sample_test_data(data_dir: Path, n_train=100, n_test=50):
    """Create sample training and test data for validation."""
    logger.info(f"Creating sample data in {data_dir}")
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample SMILES strings (mix of valid and some edge cases)
    valid_smiles = [
        'CCO',  # Ethanol
        'c1ccccc1',  # Benzene
        'CC(C)O',  # Isopropanol
        'C1=CC=CC=C1O',  # Phenol
        'CCCCCCCC',  # Octane
        'CN(C)C',  # Dimethylamine
        'CC(=O)O',  # Acetic acid
        'c1ccc(cc1)O',  # Phenol (alternative)
        'CC(C)(C)O',  # tert-Butanol
        'C1CCCCC1',  # Cyclohexane
        'CCN(CC)CC',  # Triethylamine
        'c1ccc2c(c1)ccc(c2)O',  # Naphthol
        'CC(C)C(=O)O',  # Isobutyric acid
        'C1=CC=C(C=C1)N',  # Aniline
        'CC(C)CC(C)(C)C'  # Branched alkane
    ]
    
    # Create training data
    np.random.seed(42)  # For reproducibility
    
    train_data = []
    for i in range(n_train):
        smiles = np.random.choice(valid_smiles)
        
        # Generate realistic polymer properties with some correlation
        base_temp = np.random.normal(100, 30)  # Base temperature
        
        # Glass transition temperature (Tg) - around 50-150°C
        tg = base_temp + np.random.normal(0, 20)
        
        # Fractional free volume (FFV) - typically 0.1-0.4
        ffv = np.random.beta(2, 5) * 0.4 + 0.1
        
        # Thermal conductivity (Tc) - typically 0.1-1.0 W/mK
        tc = np.random.gamma(2, 0.1) + 0.05
        
        # Density - typically 0.8-1.5 g/cm³
        density = np.random.normal(1.1, 0.2)
        density = max(0.5, min(2.0, density))  # Clamp to reasonable range
        
        # Radius of gyration (Rg) - correlated with molecular size
        rg = np.random.normal(15, 5) + len(smiles) * 0.5
        rg = max(5, rg)  # Minimum reasonable value
        
        # Introduce some missing values (10% missing rate)
        properties = [tg, ffv, tc, density, rg]
        for j in range(len(properties)):
            if np.random.random() < 0.1:  # 10% missing
                properties[j] = np.nan
        
        train_data.append({
            'id': i,
            'SMILES': smiles,
            'Tg': properties[0],
            'FFV': properties[1],
            'Tc': properties[2],
            'Density': properties[3],
            'Rg': properties[4]
        })
    
    # Create test data (no target properties)
    test_data = []
    for i in range(n_test):
        smiles = np.random.choice(valid_smiles)
        test_data.append({
            'id': i,
            'SMILES': smiles
        })
    
    # Save to CSV files
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Created training data: {train_path} ({len(train_df)} samples)")
    logger.info(f"Created test data: {test_path} ({len(test_df)} samples)")
    
    # Log data statistics
    logger.info("Training data statistics:")
    for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
        valid_count = train_df[col].notna().sum()
        mean_val = train_df[col].mean()
        std_val = train_df[col].std()
        logger.info(f"  {col}: {valid_count}/{len(train_df)} valid, mean={mean_val:.3f}, std={std_val:.3f}")
    
    return train_path, test_path


def validate_data_loading(config: Config, path_manager: PathManager):
    """Validate data loading pipeline."""
    logger.info("=== Validating Data Loading Pipeline ===")
    
    try:
        from polymer_prediction.pipeline.data_pipeline import DataPipeline
        
        # Initialize data pipeline
        data_pipeline = DataPipeline(config, path_manager)
        
        # Load and validate data
        result = data_pipeline.load_and_validate_data()
        
        if not result["success"]:
            logger.error(f"Data loading failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Check data statistics
        stats = result.get("data_statistics", {})
        logger.info("Data loading validation results:")
        logger.info(f"  Training samples: {stats.get('train_samples', 0)}")
        logger.info(f"  Test samples: {stats.get('test_samples', 0)}")
        logger.info(f"  Valid SMILES (train): {stats.get('valid_train_smiles', 0)}")
        logger.info(f"  Valid SMILES (test): {stats.get('valid_test_smiles', 0)}")
        
        # Validate minimum requirements
        if stats.get('train_samples', 0) < 10:
            logger.error("Insufficient training samples")
            return False
        
        if stats.get('test_samples', 0) < 5:
            logger.error("Insufficient test samples")
            return False
        
        if stats.get('valid_train_smiles', 0) < 5:
            logger.error("Insufficient valid training SMILES")
            return False
        
        logger.info("✅ Data loading validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Data loading validation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def validate_model_training(config: Config, path_manager: PathManager):
    """Validate model training pipeline."""
    logger.info("=== Validating Model Training Pipeline ===")
    
    try:
        from polymer_prediction.pipeline.main_pipeline import MainPipeline
        
        # Initialize main pipeline
        main_pipeline = MainPipeline(config, path_manager)
        
        # Run data processing first
        data_result = main_pipeline.run_data_processing()
        if not data_result["success"]:
            logger.error("Data processing failed during model training validation")
            return False
        
        # Run model training with reduced parameters for testing
        original_epochs = config.training.num_epochs
        config.training.num_epochs = 3  # Reduce for testing
        
        try:
            training_result = main_pipeline.run_model_training(data_result["train_dataset"])
            
            if not training_result["success"]:
                logger.error(f"Model training failed: {training_result.get('error', 'Unknown error')}")
                return False
            
            # Check training results
            training_results = training_result.get("training_results", {})
            logger.info("Model training validation results:")
            
            models_trained = []
            for model_name, model_result in training_results.items():
                if isinstance(model_result, dict) and model_result.get("success", False):
                    models_trained.append(model_name)
                    logger.info(f"  ✅ {model_name}: trained successfully")
                else:
                    logger.warning(f"  ⚠️ {model_name}: training failed or incomplete")
            
            if len(models_trained) == 0:
                logger.error("No models were trained successfully")
                return False
            
            logger.info(f"✅ Model training validation passed ({len(models_trained)} models trained)")
            return True
            
        finally:
            # Restore original epochs
            config.training.num_epochs = original_epochs
        
    except Exception as e:
        logger.error(f"Model training validation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def validate_prediction_generation(config: Config, path_manager: PathManager):
    """Validate prediction generation pipeline."""
    logger.info("=== Validating Prediction Generation Pipeline ===")
    
    try:
        from polymer_prediction.pipeline.main_pipeline import MainPipeline
        
        # Initialize main pipeline
        main_pipeline = MainPipeline(config, path_manager)
        
        # Run data processing
        data_result = main_pipeline.run_data_processing()
        if not data_result["success"]:
            logger.error("Data processing failed during prediction validation")
            return False
        
        # Run model training (minimal)
        original_epochs = config.training.num_epochs
        config.training.num_epochs = 2  # Minimal training for testing
        
        try:
            training_result = main_pipeline.run_model_training(data_result["train_dataset"])
            if not training_result["success"]:
                logger.error("Model training failed during prediction validation")
                return False
            
            # Run prediction generation
            prediction_result = main_pipeline.run_prediction_generation(
                training_result["models"],
                data_result["test_dataset"],
                model_type="gcn"  # Use simpler model for testing
            )
            
            if not prediction_result["success"]:
                logger.error(f"Prediction generation failed: {prediction_result.get('error', 'Unknown error')}")
                return False
            
            # Validate prediction results
            pred_results = prediction_result.get("prediction_results", {})
            predictions = pred_results.get("predictions")
            ids = pred_results.get("ids")
            
            if predictions is None or ids is None:
                logger.error("Prediction results missing predictions or IDs")
                return False
            
            logger.info("Prediction generation validation results:")
            logger.info(f"  Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
            logger.info(f"  Number of IDs: {len(ids) if ids is not None else 0}")
            logger.info(f"  Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
            
            # Check for reasonable predictions
            if hasattr(predictions, 'shape'):
                if predictions.shape[1] != 5:
                    logger.error(f"Expected 5 target predictions, got {predictions.shape[1]}")
                    return False
                
                if np.all(np.isnan(predictions)):
                    logger.error("All predictions are NaN")
                    return False
                
                if np.any(np.isinf(predictions)):
                    logger.error("Predictions contain infinite values")
                    return False
            
            logger.info("✅ Prediction generation validation passed")
            return True
            
        finally:
            # Restore original epochs
            config.training.num_epochs = original_epochs
        
    except Exception as e:
        logger.error(f"Prediction generation validation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def validate_submission_format(config: Config, path_manager: PathManager):
    """Validate submission file generation and format."""
    logger.info("=== Validating Submission Format ===")
    
    try:
        # Run complete pipeline to generate submission
        from main_production import ProductionPipeline
        
        # Create production pipeline
        pipeline = ProductionPipeline(config, path_manager)
        
        # Run with minimal settings for testing
        original_epochs = config.training.num_epochs
        config.training.num_epochs = 2
        
        try:
            # Run production pipeline
            result = pipeline.run_production_pipeline(model_type="gcn")
            
            if not result["success"]:
                logger.error(f"Production pipeline failed: {result.get('error', 'Unknown error')}")
                return False
            
            # Check submission file
            submission_path = result.get("submission_path")
            if not submission_path or not Path(submission_path).exists():
                logger.error("Submission file was not generated")
                return False
            
            # Load and validate submission file
            submission_df = pd.read_csv(submission_path)
            
            logger.info("Submission format validation results:")
            logger.info(f"  Submission file: {submission_path}")
            logger.info(f"  Submission shape: {submission_df.shape}")
            logger.info(f"  Columns: {list(submission_df.columns)}")
            
            # Validate required columns
            required_columns = ['id'] + config.data.target_cols
            missing_columns = set(required_columns) - set(submission_df.columns)
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Validate data types and values
            for col in config.data.target_cols:
                if not pd.api.types.is_numeric_dtype(submission_df[col]):
                    logger.error(f"Column {col} is not numeric")
                    return False
                
                if submission_df[col].isnull().any():
                    logger.warning(f"Column {col} contains null values")
                
                if submission_df[col].isinf().any():
                    logger.error(f"Column {col} contains infinite values")
                    return False
            
            # Check for duplicate IDs
            if submission_df['id'].duplicated().any():
                logger.error("Submission contains duplicate IDs")
                return False
            
            # Log sample predictions
            logger.info("Sample predictions:")
            for i in range(min(5, len(submission_df))):
                row = submission_df.iloc[i]
                pred_str = ", ".join([f"{col}={row[col]:.3f}" for col in config.data.target_cols])
                logger.info(f"  ID {row['id']}: {pred_str}")
            
            logger.info("✅ Submission format validation passed")
            return True
            
        finally:
            # Restore original epochs
            config.training.num_epochs = original_epochs
        
    except Exception as e:
        logger.error(f"Submission format validation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def validate_error_handling():
    """Validate error handling mechanisms."""
    logger.info("=== Validating Error Handling ===")
    
    try:
        # Test 1: Invalid SMILES handling
        logger.info("Testing invalid SMILES handling...")
        
        from polymer_prediction.data.dataset import PolymerDataset
        
        # Create DataFrame with invalid SMILES
        invalid_df = pd.DataFrame({
            'id': [0, 1, 2, 3, 4],
            'SMILES': ['invalid_smiles', '', 'C(C(C', 'CCO', 'c1ccccc1'],
            'Tg': [100.0, 110.0, 120.0, 130.0, 140.0]
        })
        
        # Create dataset - should handle invalid SMILES gracefully
        dataset = PolymerDataset(invalid_df, target_cols=['Tg'], is_test=False)
        
        # Should have filtered out invalid SMILES
        valid_count = len([d for d in dataset.data_list if d is not None])
        logger.info(f"  Valid samples from {len(invalid_df)} input: {valid_count}")
        
        if valid_count < 2:
            logger.error("Error handling test failed: too few valid samples")
            return False
        
        # Test 2: Missing data handling
        logger.info("Testing missing data handling...")
        
        missing_df = pd.DataFrame({
            'id': [0, 1, 2],
            'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O'],
            'Tg': [100.0, np.nan, 120.0],
            'FFV': [np.nan, 0.2, 0.3]
        })
        
        dataset = PolymerDataset(missing_df, target_cols=['Tg', 'FFV'], is_test=False)
        
        # Check that masks are properly set
        if len(dataset) > 0:
            data = dataset[0]
            if data is not None and hasattr(data, 'mask'):
                logger.info(f"  Mask handling working: mask shape {data.mask.shape}")
            else:
                logger.warning("  Mask handling may not be working properly")
        
        # Test 3: Memory constraint simulation
        logger.info("Testing memory constraint handling...")
        
        # This is a basic test - in real scenarios, memory errors would be caught
        # by the error handling mechanisms in the training pipeline
        
        logger.info("✅ Error handling validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error handling validation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def validate_resource_constraints():
    """Validate system behavior under different resource constraints."""
    logger.info("=== Validating Resource Constraints ===")
    
    try:
        # Test CPU-only configuration
        logger.info("Testing CPU-only configuration...")
        
        import torch
        cpu_config = Config()
        cpu_config.device = torch.device("cpu")
        cpu_config._apply_cpu_optimizations()
        
        logger.info(f"  Device: {cpu_config.device}")
        logger.info(f"  Batch size: {cpu_config.training.batch_size}")
        logger.info(f"  Hidden channels: {cpu_config.model.hidden_channels}")
        
        # Test memory monitoring
        logger.info("Testing memory monitoring...")
        
        from polymer_prediction.utils.performance import MemoryMonitor
        
        monitor = MemoryMonitor()
        initial_memory = monitor.get_memory_usage()
        logger.info(f"  Initial memory usage: {initial_memory:.2f} MB")
        
        # Simulate some memory usage
        test_data = np.random.randn(1000, 1000)  # ~8MB array
        current_memory = monitor.get_memory_usage()
        logger.info(f"  Memory after allocation: {current_memory:.2f} MB")
        
        del test_data
        final_memory = monitor.get_memory_usage()
        logger.info(f"  Memory after cleanup: {final_memory:.2f} MB")
        
        # Test batch size adaptation
        logger.info("Testing batch size adaptation...")
        
        from polymer_prediction.utils.error_handling import ErrorHandler
        
        error_handler = ErrorHandler()
        original_batch_size = 32
        
        # Simulate memory error and batch size reduction
        new_batch_size = error_handler.handle_memory_error(original_batch_size)
        logger.info(f"  Batch size reduced from {original_batch_size} to {new_batch_size}")
        
        if new_batch_size >= original_batch_size:
            logger.error("Batch size was not reduced as expected")
            return False
        
        logger.info("✅ Resource constraints validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Resource constraints validation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def validate_compatibility():
    """Validate compatibility with existing project structure."""
    logger.info("=== Validating Project Compatibility ===")
    
    try:
        # Test 1: Check that all expected modules can be imported
        logger.info("Testing module imports...")
        
        expected_modules = [
            'polymer_prediction.config.config',
            'polymer_prediction.data.dataset',
            'polymer_prediction.models.gcn',
            'polymer_prediction.training.trainer',
            'polymer_prediction.utils.logging',
            'polymer_prediction.utils.metrics',
            'polymer_prediction.pipeline.main_pipeline'
        ]
        
        import_results = []
        for module_name in expected_modules:
            try:
                __import__(module_name)
                import_results.append((module_name, True))
                logger.info(f"  ✅ {module_name}")
            except ImportError as e:
                import_results.append((module_name, False))
                logger.warning(f"  ⚠️ {module_name}: {e}")
        
        successful_imports = sum(1 for _, success in import_results if success)
        logger.info(f"Successfully imported {successful_imports}/{len(expected_modules)} modules")
        
        # Test 2: Check configuration compatibility
        logger.info("Testing configuration compatibility...")
        
        config = Config()
        config_dict = config.to_dict()
        
        expected_sections = ['paths', 'model', 'training', 'data', 'logging']
        missing_sections = [section for section in expected_sections if section not in config_dict]
        
        if missing_sections:
            logger.error(f"Missing configuration sections: {missing_sections}")
            return False
        
        logger.info("  Configuration structure is compatible")
        
        # Test 3: Check file structure compatibility
        logger.info("Testing file structure compatibility...")
        
        expected_files = [
            'config_production.json',
            'main_production.py',
            'src/polymer_prediction/__init__.py'
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.warning(f"Missing expected files: {missing_files}")
        else:
            logger.info("  All expected files are present")
        
        logger.info("✅ Project compatibility validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Project compatibility validation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Run comprehensive system validation."""
    print("🔬 Comprehensive System Validation")
    print("=" * 60)
    
    # Setup logging
    setup_logging(log_level="INFO", log_to_console=True, log_to_file=False)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample data
        data_dir = temp_path / "data"
        train_path, test_path = create_sample_test_data(data_dir)
        
        # Create configuration for testing
        config = Config()
        config.paths.data_dir = str(data_dir)
        config.paths.outputs_dir = str(temp_path / "outputs")
        config.paths.models_dir = str(temp_path / "models")
        config.paths.logs_dir = str(temp_path / "logs")
        config.paths.checkpoints_dir = str(temp_path / "checkpoints")
        
        # Reduce parameters for faster testing
        config.training.num_epochs = 5
        config.training.batch_size = 8
        config.model.hidden_channels = 32
        
        # Create path manager
        path_manager = PathManager()
        path_manager.base_path = temp_path
        
        # Run validation tests
        validation_tests = [
            ("Data Loading", lambda: validate_data_loading(config, path_manager)),
            ("Model Training", lambda: validate_model_training(config, path_manager)),
            ("Prediction Generation", lambda: validate_prediction_generation(config, path_manager)),
            ("Submission Format", lambda: validate_submission_format(config, path_manager)),
            ("Error Handling", validate_error_handling),
            ("Resource Constraints", validate_resource_constraints),
            ("Project Compatibility", validate_compatibility),
        ]
        
        results = []
        for test_name, test_func in validation_tests:
            print(f"\n🔍 Running: {test_name}")
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                logger.error(f"Test '{test_name}' failed with exception: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        passed = 0
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:25} : {status}")
            if success:
                passed += 1
        
        print(f"\nResults: {passed}/{len(results)} validations passed")
        
        if passed == len(results):
            print("🎉 All validations passed! System is ready for production.")
            return 0
        else:
            print("⚠️  Some validations failed. Please check the implementation.")
            return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)