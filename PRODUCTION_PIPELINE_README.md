# Production-Ready Polymer Prediction Pipeline

This document describes the production-ready main pipeline for the polymer prediction system, which integrates all fixed components and new functionality with comprehensive error handling, logging, and configuration management.

## Overview

The production pipeline (`main_production.py`) is a complete, robust implementation that provides:

- **Command-line argument parsing** for configurable training parameters and output paths
- **Comprehensive logging and progress reporting** throughout the entire pipeline
- **Model checkpointing and saving** for reproducibility and deployment
- **Submission file generation with validation** of output format
- **Integration of all fixed components** and new functionality
- **Robust error handling and recovery mechanisms**

## Features

### 🚀 Core Features

- **Modular Architecture**: Built on top of the existing modular pipeline structure
- **Configuration Management**: Centralized configuration with JSON file support and command-line overrides
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux
- **Device Auto-Detection**: Automatically detects and optimizes for CPU/GPU
- **Memory Management**: Intelligent memory monitoring and optimization
- **Progress Tracking**: Detailed progress reporting with performance metrics

### 🔧 Production Features

- **Checkpointing**: Automatic checkpointing at each pipeline stage
- **Model Artifacts**: Comprehensive model saving for reproducibility
- **Submission Validation**: Automatic validation of submission file format
- **Comprehensive Reporting**: Detailed JSON reports with all pipeline metrics
- **Error Recovery**: Graceful error handling with detailed error reporting
- **Logging**: Structured logging with multiple output formats

## Quick Start

### Basic Usage

```bash
# Run with default configuration
python main_production.py

# Run with custom configuration file
python main_production.py --config config_production.json

# Run with specific model type
python main_production.py --model-type stacking_ensemble
```

### Advanced Usage

```bash
# Run with custom parameters
python main_production.py \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --log-level DEBUG \
    --output-dir my_outputs

# Run with performance optimizations
python main_production.py \
    --enable-caching \
    --memory-monitoring \
    --force-cpu \
    --structured-logging

# Dry run to validate configuration
python main_production.py --dry-run --debug
```

## Command-Line Arguments

### Configuration Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | None | Path to configuration file (JSON format) |
| `--model-type` | str | stacking_ensemble | Type of model (gcn, tree_ensemble, stacking_ensemble) |

### Data Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | str | info | Directory containing training and test data files |
| `--output-dir` | str | outputs | Directory for output files |
| `--submission-filename` | str | auto | Custom filename for submission file |

### Training Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 50 | Number of training epochs |
| `--batch-size` | int | 32 | Training batch size |
| `--learning-rate` | float | 0.001 | Learning rate |
| `--hidden-channels` | int | 128 | Number of hidden channels in GCN |
| `--num-gcn-layers` | int | 3 | Number of GCN layers |
| `--n-folds` | int | 5 | Number of cross-validation folds |
| `--random-seed` | int | 42 | Random seed for reproducibility |

### Logging Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--log-level` | str | INFO | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--log-file` | str | auto | Log file name |
| `--no-console-log` | flag | False | Disable console logging |
| `--structured-logging` | flag | True | Enable structured JSON logging |

### Performance Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--force-cpu` | flag | False | Force CPU-only training |
| `--enable-caching` | flag | True | Enable graph caching |
| `--memory-monitoring` | flag | True | Enable memory usage monitoring |

### Development Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--debug` | flag | False | Enable debug mode |
| `--dry-run` | flag | False | Validate configuration without training |
| `--resume-from-checkpoint` | str | None | Resume from specific checkpoint |
| `--save-checkpoints` | flag | True | Save checkpoints during execution |

## Configuration File

The pipeline supports JSON configuration files for complex setups. See `config_production.json` for a complete example.

### Configuration Structure

```json
{
  "paths": {
    "data_dir": "info",
    "outputs_dir": "outputs",
    "logs_dir": "logs"
  },
  "model": {
    "hidden_channels": 128,
    "num_gcn_layers": 3,
    "tree_models": ["lgbm", "xgb", "catboost"]
  },
  "training": {
    "num_epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "logging": {
    "level": "INFO",
    "use_structured_logging": true
  }
}
```

## Pipeline Stages

The production pipeline consists of four main stages:

### 1. Data Processing
- Load and validate training and test data
- Process SMILES strings and create molecular graphs
- Generate data statistics and validation reports
- **Checkpoint**: `data_processing_checkpoint.json`

### 2. Model Training
- Train GCN model with robust error handling
- Train tree ensemble models (LightGBM, XGBoost, CatBoost)
- Perform hyperparameter optimization
- Create stacking ensemble with cross-validation
- **Checkpoint**: `model_training_checkpoint.json`
- **Artifacts**: Model files saved in `models/training/`

### 3. Prediction Generation
- Generate predictions using trained models
- Combine predictions using ensemble methods
- Validate prediction quality and format
- **Checkpoint**: `prediction_generation_checkpoint.json`

### 4. Submission Generation
- Create submission file with proper formatting
- Validate submission file format
- Generate comprehensive pipeline report
- **Output**: Submission CSV file and JSON report

## Output Files

The pipeline generates several output files:

### Submission Files
- `submission_production_YYYYMMDD_HHMMSS.csv` - Main submission file
- Format validated against competition requirements

### Reports
- `production_report_YYYYMMDD_HHMMSS.json` - Comprehensive pipeline report
- `production_config.json` - Effective configuration used

### Logs
- `production_pipeline_YYYYMMDD_HHMMSS.log` - Detailed execution log
- Structured JSON format for easy parsing

### Checkpoints
- `data_processing_checkpoint.json` - Data processing state
- `model_training_checkpoint.json` - Model training state
- `prediction_generation_checkpoint.json` - Prediction generation state

### Model Artifacts
- `models/training/gcn_model.pt` - Trained GCN model
- `models/training/lgbm_model.model` - LightGBM model
- `models/training/xgb_model.model` - XGBoost model
- `models/training/catboost_model.model` - CatBoost model
- `models/training/stacking_ensemble.pkl` - Stacking ensemble meta-model

## Error Handling

The pipeline includes comprehensive error handling:

### Automatic Recovery
- Batch size reduction on memory errors
- Device fallback (GPU → CPU)
- Model checkpoint recovery
- Graceful degradation on component failures

### Error Reporting
- Detailed error logs with stack traces
- Error summary in final report
- Checkpoint preservation on failures
- Recovery recommendations

### Validation
- Input data validation
- Configuration validation
- Model output validation
- Submission format validation

## Performance Optimization

### Memory Management
- Automatic memory monitoring
- Garbage collection between stages
- Batch size optimization
- Graph caching for repeated operations

### CPU Optimization
- Reduced model complexity for CPU training
- Optimized batch sizes
- Efficient data loading
- Progress tracking with minimal overhead

### Caching
- Molecular graph caching
- Feature caching
- Model checkpoint caching
- Configurable cache limits

## Monitoring and Logging

### Structured Logging
- JSON format for easy parsing
- Hierarchical log levels
- Performance metrics
- Memory usage tracking

### Progress Reporting
- Real-time progress updates
- Stage completion notifications
- Performance benchmarks
- Error rate monitoring

### Metrics Collection
- Training metrics (loss, accuracy)
- Performance metrics (timing, memory)
- Data quality metrics
- Model performance metrics

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Use `--force-cpu` for CPU-only training
   - Reduce `--batch-size`
   - Enable `--memory-monitoring`

2. **Configuration Errors**
   - Use `--dry-run` to validate configuration
   - Check configuration file format
   - Verify file paths exist

3. **Data Issues**
   - Check data file format and location
   - Verify SMILES string validity
   - Review data statistics in logs

4. **Model Training Failures**
   - Check device compatibility
   - Review hyperparameter settings
   - Enable `--debug` for detailed logs

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
python main_production.py --debug --dry-run
```

This will:
- Set log level to DEBUG
- Enable console logging
- Validate configuration
- Show detailed environment information

## Examples

See `run_production_example.py` for comprehensive usage examples:

```bash
python run_production_example.py
```

This script demonstrates:
- Basic usage patterns
- Configuration file usage
- Parameter overrides
- Performance optimization
- Error handling scenarios

## Integration with Existing Components

The production pipeline integrates with all existing components:

- **Data Pipeline**: `src/polymer_prediction/pipeline/data_pipeline.py`
- **Training Pipeline**: `src/polymer_prediction/pipeline/training_pipeline.py`
- **Prediction Pipeline**: `src/polymer_prediction/pipeline/prediction_pipeline.py`
- **Configuration**: `src/polymer_prediction/config/config.py`
- **Logging**: `src/polymer_prediction/utils/logging.py`
- **Path Management**: `src/polymer_prediction/utils/path_manager.py`
- **Error Handling**: `src/polymer_prediction/utils/error_handling.py`

## Requirements

The pipeline requires all dependencies from the existing project:

- Python 3.8+
- PyTorch and PyTorch Geometric
- scikit-learn
- pandas, numpy
- LightGBM, XGBoost, CatBoost
- loguru (for enhanced logging)
- All other dependencies from `requirements.txt`

## License

This production pipeline is part of the polymer prediction project and follows the same license terms.