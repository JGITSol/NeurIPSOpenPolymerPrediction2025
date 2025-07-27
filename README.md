# NeurIPS Open Polymer Prediction Challenge 2025

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a **production-ready, industry-standard** machine learning pipeline for the **NeurIPS Open Polymer Prediction Challenge 2025**. The challenge involves predicting five key polymer properties (Tg, FFV, Tc, Density, Rg) from SMILES strings using Graph Neural Networks.

## 🏆 Challenge Overview

The NeurIPS Open Polymer Prediction 2025 challenge requires predicting polymer properties to accelerate sustainable materials research. The task involves:

- **Input**: Polymer SMILES strings
- **Output**: 5 properties - Glass transition temperature (Tg), Fractional free volume (FFV), Thermal conductivity (Tc), Density, and Radius of gyration (Rg)
- **Evaluation**: Weighted Mean Absolute Error (wMAE) with property-specific reweighting
- **Data**: 7,973 training samples with significant missing values (11.8% - 93.6% per property)

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torch-geometric rdkit pandas numpy scikit-learn tqdm
```

### Competition Data Setup

1. **Download competition data** from Kaggle and place in `info/` folder:
   - `info/train.csv` - Training data with SMILES and target properties
   - `info/test.csv` - Test data with SMILES only
   - `info/sample_submission.csv` - Sample submission format

### Training and Submission

1. **Quick training and submission:**
   ```bash
   python neurips_competition.py --epochs 50 --batch_size 32 --output submission.csv
   ```

2. **Optimized training with better hyperparameters:**
   ```bash
   python train_final_model.py
   ```

3. **Custom hyperparameters:**
   ```bash
   python neurips_competition.py --epochs 100 --batch_size 64 --hidden_channels 256 --num_layers 4 --lr 0.001
   ```

## 🏗️ Architecture

```
├── .github/                    # CI/CD workflows
├── .vscode/                    # VSCode settings
├── configs/                    # Hydra configuration files
├── data/
│   ├── processed/              # Processed data
│   └── raw/                    # Raw data
├── docs/                       # Documentation
├── models/                     # Trained models
├── notebooks/                  # Jupyter notebooks
├── reports/                    # Reports and figures
├── scripts/                    # Utility scripts
├── src/polymer_prediction/     # Main source code
├── tests/                      # Test suite
├── .dockerignore               # Docker ignore file
├── .gitignore                  # Git ignore file
├── .pre-commit-config.yaml     # Pre-commit hooks configuration
├── CHANGELOG.md                # Changelog
├── CONTRIBUTING.md             # Contribution guidelines
├── Dockerfile                  # Dockerfile
├── LICENSE                     # License
├── Makefile                    # Makefile
├── README.md                   # README
├── pyproject.toml              # Project metadata and dependencies
└── setup.cfg                   # Setup configuration
```

This diagram provides a high-level overview of the repository structure. For more details, refer to the respective directories.

## 🎯 Key Features

### 🧬 Competition-Specific
- **Multi-target Prediction**: Simultaneous prediction of 5 polymer properties
- **Missing Value Handling**: Robust handling of sparse training data
- **Competition Metrics**: Implementation of weighted MAE evaluation metric
- **SMILES Processing**: Advanced molecular featurization with RDKit
- **Graph Neural Networks**: GCN-based architecture optimized for molecular data

### 🏭 Production-Ready
- **Modular Design**: Clean separation of concerns with proper abstractions
- **Error Handling**: Comprehensive error handling for invalid SMILES
- **Reproducibility**: Deterministic training with seed management
- **Scalable Architecture**: Efficient batching and GPU support
- **Industry Standards**: Following Python best practices and conventions

### 🔬 Research-Friendly
- **Flexible Configuration**: Easy hyperparameter tuning
- **Extensible Models**: Simple to add new GNN architectures
- **Comprehensive Metrics**: Per-property and competition-specific evaluation
- **Visualization Support**: Training curves and prediction analysis

## 📊 Competition Implementation

### Multi-Target Architecture

The model predicts all 5 properties simultaneously using a shared GCN encoder:

```python
# Model outputs 5 values: [Tg, FFV, Tc, Density, Rg]
predictions = model(molecular_graph)  # Shape: (batch_size, 5)
```

### Missing Value Handling

Training data has significant missing values (11.8% - 93.6% per property). Our implementation:

- Uses binary masks to track missing values
- Computes loss only on available targets
- Handles sparse gradients efficiently

```python
# Masked loss computation
loss = masked_mse_loss(predictions, targets, masks)
```

### Competition Metric

Implements the official weighted MAE metric:

```python
from polymer_prediction.utils.competition_metrics import weighted_mae

# Calculate competition score
wmae = weighted_mae(predictions, targets, masks)
```

### Usage Examples

```bash
# Quick training (10 epochs)
python neurips_competition.py --epochs 10 --batch_size 32

# Production training with optimized hyperparameters
python train_final_model.py

# Custom configuration
python neurips_competition.py \
  --epochs 100 \
  --batch_size 64 \
  --hidden_channels 256 \
  --num_layers 4 \
  --lr 0.001 \
  --output my_submission.csv
```

## 🧬 Molecular Featurization

Our pipeline converts SMILES strings to rich graph representations:

- **Atom Features**: Atomic number, degree, hybridization, aromaticity, chirality
- **Bond Features**: Bond type, ring membership, conjugation
- **Graph Structure**: Molecular connectivity with PyTorch Geometric
- **Validation**: Automatic SMILES validation and error handling

## 📈 Model Architecture

- **Graph Neural Networks**: GCN, GAT, GraphSAGE support
- **Molecular Pooling**: Global mean/max/attention pooling
- **Regularization**: Dropout, batch normalization, weight decay
- **Optimization**: Adam, AdamW with learning rate scheduling

## 🔍 Evaluation Metrics

Comprehensive evaluation with:
- **Regression**: RMSE, MAE, R², MAPE, SMAPE
- **Classification**: Accuracy, Precision, Recall, F1, AUC
- **Visualization**: Prediction plots, training curves, molecular structures

## 🐳 Docker Support

```bash
# Build and run with Docker Compose
docker-compose up jupyter      # Jupyter Lab environment
docker-compose up tensorboard # TensorBoard visualization

# Or build manually
docker build -t polymer-prediction .
docker run -it --rm -v $(pwd):/workspace polymer-prediction
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/ -k "not slow"  # Skip slow tests

# Generate coverage report
make test && open htmlcov/index.html
```

## 📋 Available Commands

```bash
make help                    # Show all available commands
make install                 # Install package
make install-dev            # Install with dev dependencies
make test                   # Run tests with coverage
make lint                   # Run code quality checks
make format                 # Format code with black/isort
make type-check             # Run mypy type checking
make security               # Run security scans
make docs                   # Build documentation
make clean                  # Clean build artifacts
make docker-build           # Build Docker image
make train                  # Train model with default config
make hyperparameter-sweep   # Run parameter optimization
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks: `make lint type-check security test`
5. Submit a pull request

## 📚 Documentation

- **API Reference**: [https://[YOUR_USERNAME].github.io/[YOUR_REPOSITORY]/](https://[YOUR_USERNAME].github.io/[YOUR_REPOSITORY]/)
- **Examples**: See `docs/examples/` for detailed tutorials
- **Configuration**: See `configs/` for configuration options

## 🏆 Industry Standards Compliance

This project implements enterprise-grade standards:

- ✅ **PEP 8** code style with Black formatting
- ✅ **Type hints** throughout codebase
- ✅ **Comprehensive testing** with pytest
- ✅ **Security scanning** with Bandit
- ✅ **Dependency management** with modern pyproject.toml
- ✅ **CI/CD pipeline** with GitHub Actions
- ✅ **Documentation** with Sphinx
- ✅ **Containerization** with Docker
- ✅ **Configuration management** with Hydra
- ✅ **Logging** with structured logging
- ✅ **Error handling** and validation
- ✅ **Reproducibility** with seed management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NeurIPS Open Polymer Prediction Challenge organizers
- PyTorch Geometric team for excellent graph ML tools
- RDKit developers for cheminformatics utilities
- Open source community for the amazing Python ecosystem
