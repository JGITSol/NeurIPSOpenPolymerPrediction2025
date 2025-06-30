"""Configuration settings for the polymer prediction project."""

import torch


class Config:
    """Configuration class for model hyperparameters and settings."""

    def __init__(self):
        """Initialize configuration with default values."""
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model hyperparameters
        self.NUM_EPOCHS = 50
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-5
        self.HIDDEN_CHANNELS = 128
        self.NUM_GCN_LAYERS = 3
        
        # Data
        self.TEST_SPLIT_FRACTION = 0.2
        self.SEED = 42
        
        # Paths
        self.DATA_DIR = "data"
        self.MODEL_SAVE_DIR = "models"
        self.RESULTS_DIR = "results"


# Create a default configuration instance
CONFIG = Config()
