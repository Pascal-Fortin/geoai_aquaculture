# Aquaculture Machine Learning Framework

A production-quality machine learning framework for the aquaculture competition, featuring a purpose-built feature engineering package.

## Overview

This framework provides a complete machine learning pipeline for processing Sentinel-1/2 satellite time series data to classify aquaculture ponds. Both the machine learning framework and the accompanying feature engineering package (`aquaculture`) were developed specifically for this competition to handle:

- Competition-specific masking simulation
- Feature engineering (optical indices, SAR features, cross-sensor features)
- Temporal statistics
- Metadata generation
- Feature selection for model optimization and interpretability

## Documentation

Detailed documentation is available in the `docs/` directory:
- **docs/architecture/training_pipeline.md** - Comprehensive guide to the machine learning pipeline architecture, execution flow, observation process handling, and implementation details

This documentation is essential for understanding the sophisticated components of this system, including the Stratified K-Fold Cross-Validation implementation for Optuna optimization and the observation process simulation that is critical to the competition.

## Features

- **Modular Design**: Separation of concerns with specialized components
- **Multiple Model Support**: LightGBM, CatBoost, and XGBoost through a unified interface
- **Automatic Class Handling**: Built-in support for imbalanced datasets
- **Robust Validation**: Proper handling of the observation process (stochastic training vs. fixed validation)
- **Stratified K-Fold Cross-Validation**: Integrated Optuna support with cross-validation for robust hyperparameter optimization
- **Hold-out Test Set**: Separate test set for final unbiased evaluation
- **Experiment Tracking**: Reproducible experiments with automatic artifact saving
- **Comprehensive Evaluation**: Competition metric (0.6×F1 + 0.4×ROC-AUC) and standard metrics
- **Model Interpretation**: SHAP values, feature importance, and visualization tools
- **Directional Vote Features**: Novel temporal features based on the directional interaction of water and vegetation indices (MNDWI, NDWI, NDRE2, NDVI) that are robust to distribution shifts
- **Production Ready**: Type hints, documentation, logging, and error handling

### Feature Selection

The aquaculture package includes flexible feature selection utilities that allow you to:

- Select features by groups (temporal, metadata, optical, SAR, cross-sensor)
- Select specific features by name
- Select features using regex/wildcard patterns
- Select features by index position
- Use custom selection functions
- Combine multiple selection criteria with include/exclude logic
- Preserve the original feature engineering pipeline while selecting subsets for modeling

These capabilities enable efficient model training, improved interpretability, and reduced overfitting by working with relevant feature subsets while maintaining access to the complete feature set for analysis.

## Project Structure

```
geoai_aquaculture/
├── aquaculture/                 # Purpose-built feature engineering package for the competition
├── docs/                        # Documentation
│   └── architecture/
│       └── training_pipeline.md # Detailed pipeline architecture documentation
├── src/                         # Main framework source code
│   ├── config.py                # Configuration management
│   ├── model_factory.py         # Model creation with class weighting
│   ├── metrics.py               # Evaluation metrics
│   ├── evaluate.py              # Cross-validation and evaluation utilities
│   ├── optuna_utils.py          # Optuna integration with CV
│   ├── trainer.py               # Main training orchestrator
│   ├── io.py                    # Input/output utilities
│   └── plotting.py              # Visualization functions
├── notebooks/                   # Jupyter notebooks for usage examples
│   ├── 01_train_model.ipynb     # Training pipeline with Stratified K-Fold CV
│   ├── 02_model_analysis.ipynb  # Model interpretation and analysis
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
3. The `aquaculture` package is included in this repository - no separate installation needed

## Usage

See the Jupyter notebooks in the `notebooks/` directory for step-by-step tutorials:

1. **01_train_model.ipynb**: Complete training pipeline with Stratified K-Fold CV
2. **02_model_analysis.ipynb**: Model interpretation and analysis
3. **03_inference.ipynb**: Making predictions on new data

## Testing

The project includes a test suite to verify functionality. Tests are located in the `tests/` directory.

### Running Tests

To run all tests, execute the following command from the project root:

```bash
python tests/run_tests.py
```

Alternatively, you can run individual test files:

```bash
python -m unittest tests.test_logging
python -m unittest tests.test_trainer
python -m unittest tests.test_basic
```

Or use the built-in test discovery:

```bash
python -m unittest discover -s tests
```

The test suite covers:
- Configuration validation
- Model factory functionality
- Metrics calculation
- Trainer initialization and basic workflow
- Logging functionality (file and console output)

### Test Structure

- `test_basic.py`: Tests for core utilities and metrics
- `test_trainer.py`: Tests for the main training pipeline
- `test_logging.py`: Tests for the configurable logging system

## Key Features

### Stratified K-Fold Cross-Validation for Hyperparameter Optimization

The framework now uses Stratified K-Fold Cross-Validation for Optuna hyperparameter optimization instead of a single hold-out validation set:

- **Training folds**: Generate new stochastic observation realizations for every Optuna trial
- **Validation folds**: Use fixed observation realizations (pre-computed before Optuna begins)
- **Hold-out test set**: Completely separate set held out for final unbiased evaluation
- **Final model**: Trained on 100% of training data (everything except hold-out test set)
- **Evaluation**: Three-stage reporting - training (in-sample), CV (out-of-sample estimate), test (final unbiased)

### Observation Process Handling

Both the ML framework and the aquaculture feature engineering package were developed specifically for this competition to correctly implement the observation process:

- **Training folds**: Observation process resampled every Optuna trial (different realization each time)
- **Validation folds**: Fixed observation patterns throughout optimization (same realization for all trials)
- **Configurable**: Support for 1 or 5 validation realizations for averaging
- **Hold-out test set**: Processed with observation simulation but no stochasticity (deterministic)

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