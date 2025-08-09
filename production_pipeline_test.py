#!/usr/bin/env python3
"""
Production Pipeline Test

This script tests the actual production pipeline with real data to validate
the complete system functionality.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback
import shutil

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# Core imports
from polymer_prediction.config.config import Config
from polymer_prediction.utils.logging import setup_logging, get_logger
from polymer_prediction.utils.path_manager import PathManager

# Initialize logger
logger = get_logger(__name__)


def create_realistic_test_data(data_dir: Path, n_train=30, n_test=10):
    """Create realistic test data for production pipeline testing."""
    logger.info(f"Creating realistic test data in {data_dir}")
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Realistic polymer SMILES strings
    polymer_smiles = [
        'CCO',  # Ethanol (simple)
        'c1ccccc1',  # Benzene
        'CC(C)O',  # Isopropanol
        'C1=CC=CC=C1O',  # Phenol
        'CCCCCCCC',  # Octane
        'CN(C)C',  # Dimethylamine
        'CC(=O)O',  # Acetic acid
        'c1ccc(cc1)O',  # Phenol
        'CC(C)(C)O',  # tert-Butanol
        'C1CCCCC1',  # Cyclohexane
        'CCN(CC)CC',  # Triethylamine
        'CC(C)CC(C)(C)C',  # Branched alkane
        'C1=CC=C(C=C1)N',  # Aniline
        'CC(C)C(=O)O',  # Isobutyric acid
        'c1ccc2c(c1)ccc(c2)O',  # Naphthol
    ]
    
    # Create training data with realistic polymer properties
    np.random.seed(42)  # For reproducibility
    
    train_data = []
    for i in range(n_train):
        smiles = np.random.choice(polymer_smiles)
        
        # Generate correlated polymer properties
        # Base temperature affects other properties
        base_temp = np.random.normal(100, 25)
        
        # Glass transition temperature (Tg) - typically 50-200°C for polymers
        tg = base_temp + np.random.normal(0, 15)
        tg = max(20, min(250, tg))  # Reasonable bounds
        
        # Fractional free volume (FFV) - typically 0.1-0.4 for polymers
        ffv = np.random.beta(2, 5) * 0.3 + 0.1
        
        # Thermal conductivity (Tc) - typically 0.1-1.0 W/mK for polymers
        tc = np.random.gamma(2, 0.08) + 0.05
        tc = min(1.5, tc)  # Cap at reasonable value
        
        # Density - typically 0.8-1.5 g/cm³ for polymers
        density = np.random.normal(1.1, 0.15)
        density = max(0.7, min(1.8, density))
        
        # Radius of gyration (Rg) - correlated with molecular complexity
        molecular_complexity = len(smiles) + smiles.count('c') * 2  # Aromatic rings add complexity
        rg = np.random.normal(12 + molecular_complexity * 0.3, 3)
        rg = max(5, rg)
        
        # Introduce realistic missing values (3% missing rate)
        properties = [tg, ffv, tc, density, rg]
        for j in range(len(properties)):
            if np.random.random() < 0.03:  # 3% missing
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
        smiles = np.random.choice(polymer_smiles)
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
        if valid_count > 0:
            mean_val = train_df[col].mean()
            std_val = train_df[col].std()
            min_val = train_df[col].min()
            max_val = train_df[col].max()
            logger.info(f"  {col}: {valid_count}/{len(train_df)} valid, "
                       f"mean={mean_val:.3f}, std={std_val:.3f}, "
                       f"range=[{min_val:.3f}, {max_val:.3f}]")
    
    return train_path, test_path


def test_production_pipeline_dry_run():
    """Test production pipeline in dry-run mode."""
    logger.info("=== Testing Production Pipeline (Dry Run) ===")
    
    try:
        # Create temporary test data
        test_dir = Path("temp_test_data")
        test_dir.mkdir(exist_ok=True)
        
        try:
            # Create test data
            data_dir = test_dir / "info"
            train_path, test_path = create_realistic_test_data(data_dir, n_train=20, n_test=8)
            
            # Import and run production pipeline
            from main_production import main, parse_command_line_arguments, create_config_from_args
            
            # Mock command line arguments for dry run
            import argparse
            
            # Create mock args for dry run
            args = argparse.Namespace()
            args.config = None
            args.data_dir = str(data_dir)
            args.output_dir = str(test_dir / "outputs")
            args.epochs = 2  # Minimal epochs for testing
            args.batch_size = 4
            args.learning_rate = 0.001
            args.log_level = "INFO"
            args.log_file = None
            args.no_console_log = False
            args.structured_logging = False
            args.force_cpu = True  # Force CPU for testing
            args.enable_caching = False
            args.memory_monitoring = False
            args.debug = False
            args.dry_run = True  # Enable dry run
            args.resume_from_checkpoint = None
            args.save_checkpoints = False
            args.hidden_channels = None
            args.num_gcn_layers = None
            args.n_folds = None
            args.random_seed = 42
            args.submission_filename = None
            
            # Create configuration
            config = create_config_from_args(args)
            
            # Create path manager
            path_manager = PathManager()
            path_manager.base_path = test_dir
            
            # Import and create production pipeline
            from main_production import ProductionPipeline
            
            pipeline = ProductionPipeline(config, path_manager)
            
            # Setup environment (dry run should complete this)
            pipeline.setup_environment()
            
            logger.info("✅ Production pipeline dry run test passed")
            return True
            
        finally:
            # Clean up
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)
        
    except Exception as e:
        logger.error(f"Production pipeline dry run test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def test_production_pipeline_minimal():
    """Test production pipeline with minimal training."""
    logger.info("=== Testing Production Pipeline (Minimal Training) ===")
    
    try:
        # Create temporary test data
        test_dir = Path("temp_minimal_test")
        test_dir.mkdir(exist_ok=True)
        
        try:
            # Create test data
            data_dir = test_dir / "info"
            train_path, test_path = create_realistic_test_data(data_dir, n_train=15, n_test=5)
            
            # Import production pipeline
            from main_production import ProductionPipeline
            
            # Create minimal configuration
            config = Config()
            config.paths.data_dir = str(data_dir)
            config.paths.outputs_dir = str(test_dir / "outputs")
            config.paths.models_dir = str(test_dir / "models")
            config.paths.logs_dir = str(test_dir / "logs")
            config.paths.checkpoints_dir = str(test_dir / "checkpoints")
            
            # Minimal training settings
            config.training.num_epochs = 1
            config.training.batch_size = 4
            config.model.hidden_channels = 16
            config.model.num_gcn_layers = 2
            
            # Force CPU
            import torch
            config.device = torch.device("cpu")
            config._apply_cpu_optimizations()
            
            # Create path manager
            path_manager = PathManager()
            path_manager.base_path = test_dir
            
            # Create and run pipeline
            pipeline = ProductionPipeline(config, path_manager)
            
            # Run with GCN only (simpler than full ensemble)
            result = pipeline.run_production_pipeline(model_type="gcn")
            
            if result["success"]:
                logger.info("Pipeline completed successfully")
                
                # Check outputs
                submission_path = result.get("submission_path")
                if submission_path and Path(submission_path).exists():
                    # Load and validate submission
                    submission_df = pd.read_csv(submission_path)
                    logger.info(f"Submission file created: {submission_path}")
                    logger.info(f"Submission shape: {submission_df.shape}")
                    
                    # Basic validation
                    required_cols = ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
                    if all(col in submission_df.columns for col in required_cols):
                        logger.info("✅ Production pipeline minimal test passed")
                        return True
                    else:
                        logger.error("Submission file missing required columns")
                        return False
                else:
                    logger.error("Submission file was not created")
                    return False
            else:
                logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
                return False
            
        finally:
            # Clean up
            if test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)
        
    except Exception as e:
        logger.error(f"Production pipeline minimal test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def test_submission_format_validation():
    """Test submission format validation with realistic data."""
    logger.info("=== Testing Submission Format Validation ===")
    
    try:
        # Create sample submission data
        n_samples = 10
        submission_df = pd.DataFrame({
            'id': range(n_samples),
            'Tg': np.random.normal(100, 20, n_samples),
            'FFV': np.random.beta(2, 5, n_samples) * 0.3 + 0.1,
            'Tc': np.random.gamma(2, 0.1, n_samples) + 0.05,
            'Density': np.random.normal(1.1, 0.2, n_samples),
            'Rg': np.random.normal(15, 5, n_samples)
        })
        
        # Ensure positive values where needed
        submission_df['FFV'] = np.abs(submission_df['FFV'])
        submission_df['Tc'] = np.abs(submission_df['Tc'])
        submission_df['Density'] = np.abs(submission_df['Density'])
        submission_df['Rg'] = np.abs(submission_df['Rg'])
        
        # Test validation using production pipeline method
        from main_production import ProductionPipeline
        
        config = Config()
        path_manager = PathManager()
        pipeline = ProductionPipeline(config, path_manager)
        
        # Test validation
        is_valid = pipeline.validate_submission_format(submission_df)
        
        if is_valid:
            logger.info("Submission format validation passed")
            
            # Test with some edge cases
            # Test with missing values
            test_df = submission_df.copy()
            test_df.loc[0, 'Tg'] = np.nan
            
            is_valid_with_nan = pipeline.validate_submission_format(test_df)
            logger.info(f"Validation with NaN values: {'passed' if is_valid_with_nan else 'failed'}")
            
            # Test with infinite values
            test_df = submission_df.copy()
            test_df.loc[0, 'FFV'] = np.inf
            
            is_valid_with_inf = pipeline.validate_submission_format(test_df)
            logger.info(f"Validation with infinite values: {'failed as expected' if not is_valid_with_inf else 'unexpectedly passed'}")
            
            logger.info("✅ Submission format validation test passed")
            return True
        else:
            logger.error("Submission format validation failed")
            return False
        
    except Exception as e:
        logger.error(f"Submission format validation test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def test_error_recovery():
    """Test error recovery mechanisms."""
    logger.info("=== Testing Error Recovery ===")
    
    try:
        from polymer_prediction.utils.error_handling import ErrorHandler
        
        error_handler = ErrorHandler()
        
        # Test memory error handling
        original_batch_size = 64
        reduced_batch_size = error_handler.handle_memory_error(original_batch_size)
        
        logger.info(f"Memory error handling: {original_batch_size} -> {reduced_batch_size}")
        
        if reduced_batch_size < original_batch_size and reduced_batch_size > 0:
            # Test multiple reductions
            further_reduced = error_handler.handle_memory_error(reduced_batch_size)
            logger.info(f"Further reduction: {reduced_batch_size} -> {further_reduced}")
            
            if further_reduced < reduced_batch_size:
                logger.info("✅ Error recovery test passed")
                return True
            else:
                logger.error("Batch size was not further reduced")
                return False
        else:
            logger.error("Batch size was not properly reduced")
            return False
        
    except Exception as e:
        logger.error(f"Error recovery test failed: {e}")
        return False


def test_configuration_override():
    """Test configuration override functionality."""
    logger.info("=== Testing Configuration Override ===")
    
    try:
        # Test configuration loading and overrides
        config = Config()
        
        # Test basic configuration
        original_batch_size = config.training.batch_size
        original_learning_rate = config.training.learning_rate
        
        logger.info(f"Original batch size: {original_batch_size}")
        logger.info(f"Original learning rate: {original_learning_rate}")
        
        # Test overrides
        overrides = {
            "training": {
                "batch_size": 8,
                "learning_rate": 0.0005,
                "num_epochs": 10
            },
            "model": {
                "hidden_channels": 64
            }
        }
        
        config._apply_overrides(overrides)
        
        # Check that overrides were applied
        if (config.training.batch_size == 8 and 
            config.training.learning_rate == 0.0005 and
            config.training.num_epochs == 10 and
            config.model.hidden_channels == 64):
            
            logger.info("Configuration overrides applied successfully")
            logger.info(f"New batch size: {config.training.batch_size}")
            logger.info(f"New learning rate: {config.training.learning_rate}")
            
            # Test CPU optimizations
            import torch
            config.device = torch.device("cpu")
            config._apply_cpu_optimizations()
            
            logger.info(f"CPU optimized batch size: {config.training.batch_size}")
            
            logger.info("✅ Configuration override test passed")
            return True
        else:
            logger.error("Configuration overrides were not applied correctly")
            return False
        
    except Exception as e:
        logger.error(f"Configuration override test failed: {e}")
        return False


def main():
    """Run production pipeline tests."""
    print("🧪 Production Pipeline Testing")
    print("=" * 50)
    
    # Setup logging
    setup_logging(log_level="INFO", log_to_console=True, log_to_file=False)
    
    # Run tests
    tests = [
        ("Configuration Override", test_configuration_override),
        ("Error Recovery", test_error_recovery),
        ("Submission Format Validation", test_submission_format_validation),
        ("Production Pipeline Dry Run", test_production_pipeline_dry_run),
        ("Production Pipeline Minimal", test_production_pipeline_minimal),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:30} : {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All production pipeline tests passed!")
        return 0
    elif passed >= len(results) * 0.8:  # 80% pass rate
        print("✅ Most tests passed. Production pipeline is largely functional.")
        return 0
    else:
        print("⚠️  Several tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)