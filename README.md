# Aquaculture Machine Learning Framework

A production-quality machine learning framework for the aquaculture competition, built around an existing feature engineering package.

## Overview

This framework provides a complete machine learning pipeline for processing Sentinel-1/2 satellite time series data to classify aquaculture ponds. It's designed around an existing feature engineering package (`aquaculture`) that handles:

- Competition masking
- Feature engineering (optical indices, SAR features, cross-sensor features)
- Temporal statistics
- Metadata generation

## Features

- **Modular Design**: Separation of concerns with specialized components
- **Multiple Model Support**: LightGBM, CatBoost, and XGBoost through a unified interface
- **Automatic Class Handling**: Built-in support for imbalanced datasets
- **Robust Validation**: Proper handling of the observation process (stochastic training vs. fixed validation)
- **Hyperparameter Optimization**: Integrated Optuna support with pruning
- **Experiment Tracking**: Reproducible experiments with automatic artifact saving
- **Comprehensive Evaluation**: Competition metric (0.6×F1 + 0.4×ROC-AUC) and standard metrics
- **Model Interpretation**: SHAP values, feature importance, and visualization tools
- **Production Ready**: Type hints, documentation, logging, and error handling

## Project Structure

```
geoai_aquaculture/
├── aquaculture/                 # Existing feature engineering package
├── src/                         # Main framework source code
│   ├── config.py                # Configuration management
│   ├── model_factory.py         # Model creation with class weighting
│   ├── metrics.py               # Evaluation metrics
│   ├── evaluate.py              # Cross-validation and evaluation utilities
│   ├── optuna_utils.py          # Optuna integration
│   ├── trainer.py               # Main training orchestrator
│   ├── io.py                    # Input/output utilities
│   └── plotting.py              # Visualization functions
├── notebooks/                   # Jupyter notebooks for usage examples
│   ├── 01_train_model.ipynb     # Training tutorial
│   ├── 02_model_analysis.ipynb  # Model analysis tutorial
│   └── 03_inference.ipynb       # Inference tutorial
├── tests/                       # Unit tests
├── experiments/                 # Experiment outputs (created during runtime)
├── models/                      # Saved models (created during runtime)
└── requirements.txt             # Python dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the existing `aquaculture` package is available in your Python path

## Usage

See the Jupyter notebooks in the `notebooks/` directory for step-by-step tutorials:

1. **01_train_model.ipynb**: Complete training pipeline
2. **02_model_analysis.ipynb**: Model interpretation and analysis
3. **03_inference.ipynb**: Making predictions on new data

## Key Features

### Observation Process Handling
The framework correctly implements the competition's observation process:
- **Training**: Observation process resampled every Optuna trial
- **Validation**: Fixed observation patterns throughout optimization
- **Configurable**: Support for 1 or 5 validation realizations

### Model Factory
Automatically handles class imbalance:
- LightGBM: `scale_pos_weight` parameter
- CatBoost: `auto_class_weights='Balanced'`
- XGBoost: `scale_pos_weight` parameter

### Experiment Tracking
Each training run creates a timestamped directory containing:
- Configuration files
- Best model and parameters
- Feature importance and names
- Evaluation metrics
- Optuna study object
- Generated plots and explanations

## Requirements

- Python 3.11+
- See `requirements.txt` for detailed dependencies

## License

MIT