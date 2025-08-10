#!/usr/bin/env python3
"""
Quick test of the monolith with minimal data to verify it works.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Create minimal test data
train_data = {
    'id': range(50),
    'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O', 'C1=CC=CC=C1O', 'CCCCCCCC'] * 10,
    'Tg': np.random.randn(50),
    'FFV': np.random.randn(50),
    'Tc': np.random.randn(50),
    'Density': np.random.randn(50),
    'Rg': np.random.randn(50)
}

test_data = {
    'id': range(50, 55),
    'SMILES': ['CCO', 'c1ccccc1', 'CC(C)O', 'C1=CC=CC=C1O', 'CCCCCCCC']
}

# Save test data
pd.DataFrame(train_data).to_csv('test_train.csv', index=False)
pd.DataFrame(test_data).to_csv('test_test.csv', index=False)

print("Created minimal test data")

# Test the monolith with reduced parameters
import sys
sys.path.append('.')

# Modify the monolith config for quick testing
import kaggle_polymer_prediction_monolith as monolith

# Override config for quick test
monolith.config.DATA_PATH = "."
monolith.config.TRAIN_FILE = "test_train.csv"
monolith.config.TEST_FILE = "test_test.csv"
monolith.config.SUBMISSION_FILE = "test_submission.csv"
monolith.config.NUM_EPOCHS = 5  # Quick test
monolith.config.BATCH_SIZE = 8
monolith.config.HIDDEN_CHANNELS = 64
monolith.config.NUM_GCN_LAYERS = 2

print("Testing monolith with minimal data...")

try:
    monolith.main()
    print("Monolith test completed successfully!")
except Exception as e:
    print(f"Monolith test failed: {e}")
    import traceback
    traceback.print_exc()