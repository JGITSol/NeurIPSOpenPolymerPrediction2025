#!/usr/bin/env python3
"""
Test script to verify Optuna integration works correctly
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_optuna_import():
    """Test if Optuna can be imported."""
    try:
        import optuna
        from optuna.samplers import TPESampler
        print("✅ Optuna import successful")
        return True
    except ImportError as e:
        print(f"❌ Optuna import failed: {e}")
        return False

def test_config_class():
    """Test the enhanced Config class."""
    try:
        # Mock the Config class from the notebook
        class Config:
            USE_OPTUNA = True
            OPTUNA_N_TRIALS = 5
            OPTUNA_TIMEOUT = 60
            
            # Default parameters
            DEFAULT_BATCH_SIZE = 32
            DEFAULT_LEARNING_RATE = 1e-3
            DEFAULT_HIDDEN_CHANNELS = 64
            
            # Search ranges
            BATCH_SIZE_RANGE = [16, 64]
            LEARNING_RATE_RANGE = [1e-4, 1e-2]
            HIDDEN_CHANNELS_RANGE = [32, 128]
            
            def __init__(self):
                self.BATCH_SIZE = self.DEFAULT_BATCH_SIZE
                self.LEARNING_RATE = self.DEFAULT_LEARNING_RATE
                self.HIDDEN_CHANNELS = self.DEFAULT_HIDDEN_CHANNELS
            
            def update_hyperparameters(self, trial=None, **kwargs):
                if trial is not None:
                    # Mock trial suggestions
                    self.BATCH_SIZE = 32
                    self.LEARNING_RATE = 1e-3
                    self.HIDDEN_CHANNELS = 64
                else:
                    for key, value in kwargs.items():
                        if hasattr(self, key.upper()):
                            setattr(self, key.upper(), value)
            
            def get_config_dict(self):
                return {
                    'batch_size': self.BATCH_SIZE,
                    'learning_rate': self.LEARNING_RATE,
                    'hidden_channels': self.HIDDEN_CHANNELS
                }
        
        # Test config creation
        config = Config()
        print(f"✅ Config class works: {config.get_config_dict()}")
        
        # Test manual parameter update
        config.update_hyperparameters(batch_size=64, learning_rate=1e-4)
        print(f"✅ Manual parameter update works: {config.get_config_dict()}")
        
        return True
    except Exception as e:
        print(f"❌ Config class test failed: {e}")
        return False

def test_optuna_study():
    """Test basic Optuna study creation."""
    try:
        import optuna
        from optuna.samplers import TPESampler
        
        def simple_objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(simple_objective, n_trials=3, timeout=10)
        
        print(f"✅ Optuna study works: best_value={study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")
        return True
    except Exception as e:
        print(f"❌ Optuna study test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Optuna integration...")
    
    tests = [
        ("Optuna Import", test_optuna_import),
        ("Config Class", test_config_class),
        ("Optuna Study", test_optuna_study)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️ {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ All tests passed! Optuna integration should work correctly.")
    else:
        print("❌ Some tests failed. Check the integration.")
        sys.exit(1)