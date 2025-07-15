# Polymer Prediction Documentation

Welcome to the Polymer Prediction project documentation. This project provides an industry-standard machine learning pipeline for predicting polymer properties from molecular structures.

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from polymer_prediction.main import main
import argparse

# Create arguments
args = argparse.Namespace(
    data_path="data/train.csv",
    target_column="target_property",
    plot=True
)

# Run training
main(args)
```

### Using Configuration Files

```bash
# Train with default configuration
python -m polymer_prediction.main

# Train with custom configuration
python -m polymer_prediction.main --config-name=experiment/custom

# Run hyperparameter sweep
python -m polymer_prediction.main --multirun model.hidden_channels=64,128,256
```

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data, models, training, and utilities
- **Configuration Management**: Hydra-based configuration system for easy experimentation
- **Data Validation**: Comprehensive data validation and cleaning utilities
- **Multiple Model Support**: Extensible model architecture supporting various GNN models
- **Comprehensive Metrics**: Detailed evaluation metrics and visualization tools
- **Industry Standards**: Follows Python packaging, testing, and documentation best practices
- **Docker Support**: Containerized development and deployment
- **CI/CD Pipeline**: Automated testing and quality checks

## Project Structure

```
├── src/polymer_prediction/     # Main source code
│   ├── config/                 # Configuration management
│   ├── data/                   # Data loading and processing
│   ├── models/                 # Model definitions
│   ├── preprocessing/          # Data preprocessing
│   ├── training/               # Training utilities
│   ├── utils/                  # Utility functions
│   └── visualization/          # Plotting and visualization
├── tests/                      # Test suite
├── configs/                    # Configuration files
├── docs/                       # Documentation
└── scripts/                    # Utility scripts
```

## API Reference

See the [API Reference](api.md) for detailed documentation of all modules and functions.

## Examples

Check out the [Examples](examples.md) section for detailed usage examples and tutorials.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to this project.