#!/usr/bin/env python3
"""
Test script for the production-ready polymer prediction pipeline.

This script performs basic validation and testing of the production pipeline
components to ensure they work correctly.
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required imports work correctly."""
    print("Testing imports...")
    
    try:
        from polymer_prediction.config.config import Config
        from polymer_prediction.utils.logging import setup_logging, get_logger
        from polymer_prediction.utils.path_manager import PathManager
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration loading and validation."""
    print("Testing configuration...")
    
    try:
        from polymer_prediction.config.config import Config
        
        # Test default configuration
        config = Config()
        assert hasattr(config, 'device')
        assert hasattr(config, 'paths')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        
        # Test configuration serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'paths' in config_dict
        assert 'model' in config_dict
        
        # Test configuration file operations
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_config(temp_path)
            
            # Load configuration back
            loaded_config = Config.load_config(temp_path)
            assert loaded_config is not None
        finally:
            # Clean up
            try:
                Path(temp_path).unlink()
            except:
                pass
        
        print("✅ Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_path_manager():
    """Test path manager functionality."""
    print("Testing path manager...")
    
    try:
        from polymer_prediction.utils.path_manager import PathManager
        
        # Create path manager
        path_manager = PathManager()
        
        # Test path resolution
        test_path = path_manager.resolve_path("test_file.txt")
        assert isinstance(test_path, Path)
        
        # Test directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager.base_path = Path(temp_dir)
            
            # Test various path methods
            data_path = path_manager.get_data_path("test.csv")
            model_path = path_manager.get_model_path("test.pt")
            output_path = path_manager.get_output_path("test.csv")
            
            assert all(isinstance(p, Path) for p in [data_path, model_path, output_path])
        
        print("✅ Path manager tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Path manager test failed: {e}")
        return False


def test_logging():
    """Test logging functionality."""
    print("Testing logging...")
    
    try:
        from polymer_prediction.utils.logging import setup_logging, get_logger
        
        # Setup basic logging
        setup_logging(log_level="INFO", log_to_console=False, log_to_file=False)
        
        # Get logger
        logger = get_logger("test_logger")
        
        # Test logging methods
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        
        # Test context
        logger.add_context(test_key="test_value")
        logger.info("Test message with context")
        logger.clear_context()
        
        print("✅ Logging tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False


def test_command_line_parsing():
    """Test command line argument parsing."""
    print("Testing command line parsing...")
    
    try:
        # Import the main script as a module
        import importlib.util
        
        main_script_path = Path("main_production.py")
        if not main_script_path.exists():
            print("❌ Main script not found")
            return False
        
        spec = importlib.util.spec_from_file_location("main_production", main_script_path)
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Mock sys.argv to test argument parsing
        with patch('sys.argv', ['main_production.py', '--help']):
            try:
                # This should raise SystemExit due to --help
                main_module.parse_command_line_arguments()
            except SystemExit:
                pass  # Expected behavior for --help
        
        # Test with valid arguments
        with patch('sys.argv', [
            'main_production.py',
            '--model-type', 'gcn',
            '--epochs', '10',
            '--batch-size', '16',
            '--log-level', 'DEBUG'
        ]):
            args = main_module.parse_command_line_arguments()
            assert args.model_type == 'gcn'
            assert args.epochs == 10
            assert args.batch_size == 16
            assert args.log_level == 'DEBUG'
        
        print("✅ Command line parsing tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Command line parsing test failed: {e}")
        return False


def test_config_creation_from_args():
    """Test configuration creation from command line arguments."""
    print("Testing config creation from args...")
    
    try:
        import importlib.util
        
        main_script_path = Path("main_production.py")
        spec = importlib.util.spec_from_file_location("main_production", main_script_path)
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Create mock arguments
        mock_args = Mock()
        mock_args.config = None
        mock_args.data_dir = "test_data"
        mock_args.output_dir = "test_outputs"
        mock_args.epochs = 25
        mock_args.batch_size = 8
        mock_args.learning_rate = 0.0005
        mock_args.log_level = "DEBUG"
        mock_args.log_file = "test.log"
        mock_args.no_console_log = False
        mock_args.structured_logging = True
        mock_args.force_cpu = True
        mock_args.enable_caching = True
        mock_args.memory_monitoring = True
        mock_args.debug = True
        mock_args.hidden_channels = None
        mock_args.num_gcn_layers = None
        mock_args.n_folds = None
        mock_args.random_seed = None
        
        # Test config creation
        config = main_module.create_config_from_args(mock_args)
        
        # Verify overrides were applied
        assert config.paths.data_dir == "test_data"
        assert config.paths.outputs_dir == "test_outputs"
        assert config.training.num_epochs == 25
        assert config.training.batch_size == 8
        assert config.training.learning_rate == 0.0005
        assert config.logging.level == "DEBUG"
        assert config.logging.log_file == "test.log"
        
        print("✅ Config creation from args tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Config creation from args test failed: {e}")
        return False


def test_production_pipeline_class():
    """Test ProductionPipeline class initialization."""
    print("Testing ProductionPipeline class...")
    
    try:
        import importlib.util
        
        main_script_path = Path("main_production.py")
        spec = importlib.util.spec_from_file_location("main_production", main_script_path)
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        from polymer_prediction.config.config import Config
        from polymer_prediction.utils.path_manager import PathManager
        
        # Create test configuration and path manager
        config = Config()
        path_manager = PathManager()
        
        # Initialize ProductionPipeline
        pipeline = main_module.ProductionPipeline(config, path_manager)
        
        # Test basic attributes
        assert hasattr(pipeline, 'config')
        assert hasattr(pipeline, 'path_manager')
        assert hasattr(pipeline, 'error_handler')
        assert hasattr(pipeline, 'main_pipeline')
        
        # Test methods exist
        assert hasattr(pipeline, 'setup_environment')
        assert hasattr(pipeline, 'create_checkpoint')
        assert hasattr(pipeline, 'load_checkpoint')
        assert hasattr(pipeline, 'validate_submission_format')
        assert hasattr(pipeline, 'run_production_pipeline')
        
        print("✅ ProductionPipeline class tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ProductionPipeline class test failed: {e}")
        return False


def test_configuration_file():
    """Test configuration file loading."""
    print("Testing configuration file...")
    
    try:
        config_file = Path("config_production.json")
        if not config_file.exists():
            print("❌ Configuration file not found")
            return False
        
        # Load and validate configuration file
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Check required sections
        required_sections = ['paths', 'model', 'training', 'data', 'logging', 'performance']
        for section in required_sections:
            assert section in config_data, f"Missing section: {section}"
        
        # Check some key values
        assert 'target_cols' in config_data['data']
        assert isinstance(config_data['data']['target_cols'], list)
        assert len(config_data['data']['target_cols']) == 5
        
        print("✅ Configuration file tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration file test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Production Pipeline Components")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Path Manager", test_path_manager),
        ("Logging", test_logging),
        ("Command Line Parsing", test_command_line_parsing),
        ("Config from Args", test_config_creation_from_args),
        ("ProductionPipeline Class", test_production_pipeline_class),
        ("Configuration File", test_configuration_file),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:25} : {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Production pipeline is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)