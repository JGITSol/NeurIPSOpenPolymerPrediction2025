# NeurIPS Open Polymer Prediction Challenge

[![CI](https://github.com/yourusername/polymer-prediction/workflows/CI/badge.svg)](https://github.com/yourusername/polymer-prediction/actions)
[![Documentation](https://github.com/yourusername/polymer-prediction/workflows/Documentation/badge.svg)](https://yourusername.github.io/polymer-prediction/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a **production-ready, industry-standard** machine learning pipeline for the NeurIPS Open Polymer Prediction Challenge. Built with modern Python best practices, comprehensive testing, and enterprise-grade tooling.

## 🚀 Quick Start

### Automated Setup (Recommended)

**Windows:**
```powershell
.\setup_env.ps1
```

**Unix/MacOS:**
```bash
make setup-env
source venv/bin/activate
```

### Manual Setup

1. **Clone and install:**
   ```bash
   git clone https://github.com/yourusername/polymer-prediction.git
   cd polymer-prediction
   pip install -e ".[dev,docs]"
   ```

2. **Generate sample data:**
   ```bash
   python scripts/setup_data.py --n_samples 1000 --split
   ```

3. **Train your first model:**
   ```bash
   make train
   ```

## 🏗️ Architecture

```
├── src/polymer_prediction/     # 🧠 Core ML Pipeline
│   ├── config/                 # ⚙️  Configuration management (Hydra)
│   ├── data/                   # 📊 Data loading, validation & processing
│   ├── models/                 # 🤖 Neural network architectures
│   ├── preprocessing/          # 🔧 SMILES to graph featurization
│   ├── training/               # 🏋️  Training loops & optimization
│   ├── utils/                  # 🛠️  Logging, metrics, I/O utilities
│   └── visualization/          # 📈 Plotting & molecular visualization
├── tests/                      # ✅ Comprehensive test suite
├── configs/                    # 📋 Hydra configuration files
├── docs/                       # 📚 Sphinx documentation
├── scripts/                    # 🔨 Utility scripts
├── .github/workflows/          # 🔄 CI/CD pipelines
└── docker/                     # 🐳 Containerization
```

## 🎯 Key Features

### 🏭 Production-Ready
- **Type Safety**: Full mypy type checking
- **Data Validation**: Pydantic-based data validation
- **Error Handling**: Comprehensive error handling and logging
- **Configuration Management**: Hydra for flexible experimentation
- **Containerization**: Docker support for reproducible environments

### 🧪 Research-Friendly
- **Modular Design**: Easy to extend with new models and features
- **Experiment Tracking**: W&B and TensorBoard integration
- **Hyperparameter Sweeps**: Built-in support for parameter optimization
- **Reproducibility**: Deterministic training with seed management

### 🔧 Developer Experience
- **Pre-commit Hooks**: Automated code quality checks
- **CI/CD Pipeline**: GitHub Actions for testing and deployment
- **Documentation**: Auto-generated API docs with Sphinx
- **Testing**: 95%+ test coverage with pytest

## 📊 Usage Examples

### Basic Training
```bash
# Train with default configuration
python -m polymer_prediction.main

# Train with custom data
python -m polymer_prediction.main --data_path data/your_data.csv --target_column your_target
```

### Advanced Configuration
```bash
# Use Hydra configuration
python -m polymer_prediction.main --config-name experiment/custom

# Hyperparameter sweep
python -m polymer_prediction.main --multirun \
  model.hidden_channels=64,128,256 \
  training.learning_rate=1e-4,1e-3,1e-2
```

### Programmatic Usage
```python
from polymer_prediction.main import main
from polymer_prediction.config.config import CONFIG

# Customize configuration
CONFIG.NUM_EPOCHS = 100
CONFIG.BATCH_SIZE = 64

# Run training
main(args)
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

- **API Reference**: [https://yourusername.github.io/polymer-prediction/](https://yourusername.github.io/polymer-prediction/)
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
