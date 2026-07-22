"""
Basic tests for the aquaculture machine learning framework.
"""

import sys
import os
# Add the project root to the Python path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from unittest.mock import patch
from sklearn.datasets import make_classification

from src.config import TrainingConfig
from src.model_factory import ModelFactory
from src.metrics import competition_score, calculate_metrics
from src.trainer import Trainer


def test_model_factory():
    """Test that ModelFactory creates models correctly."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Test LightGBM
    lgb_model = ModelFactory.create('lightgbm', random_state=42, y_train=y)
    assert lgb_model is not None
    assert hasattr(lgb_model, 'fit')
    assert hasattr(lgb_model, 'predict')
    assert hasattr(lgb_model, 'predict_proba')

    # Test CatBoost
    cb_model = ModelFactory.create('catboost', random_state=42, y_train=y)
    assert cb_model is not None
    assert hasattr(cb_model, 'fit')
    assert hasattr(cb_model, 'predict')
    assert hasattr(cb_model, 'predict_proba')

    # Test XGBoost
    xgb_model = ModelFactory.create('xgboost', random_state=42, y_train=y)
    assert xgb_model is not None
    assert hasattr(xgb_model, 'fit')
    assert hasattr(xgb_model, 'predict')
    assert hasattr(xgb_model, 'predict_proba')


def test_competition_score():
    """Test the competition score calculation."""
    # Perfect predictions
    y_true = np.array([0, 0, 1, 1])
    y_prob_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    score_perfect = competition_score(y_true, y_prob_perfect)
    assert score_perfect == 1.0  # Should be perfect

    # Completely wrong predictions
    y_prob_wrong = np.array([0.9, 0.8, 0.2, 0.1])
    score_wrong = competition_score(y_true, y_prob_wrong)
    assert score_wrong == 0.0  # Should be zero

    # Random predictions
    y_prob_random = np.array([0.5, 0.5, 0.5, 0.5])
    score_random = competition_score(y_true, y_prob_random)
    # Should be around 0.5 (0.5 (0.6*0.5 + 0.4*0.5 = 0.5)
    assert 0.4 <= score_random <= 0.6


def test_calculate_metrics():
    """Test the comprehensive metrics calculation."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])

    metrics = calculate_metrics(y_true, y_prob)

    # Check that all expected keys are present
    expected_keys = ['competition_score', 'f1', 'roc_auc', 'precision', 'recall', 'accuracy', 'brier_score']
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], (int, float))
        assert 0 <= metrics[key] <= 1 or key == 'brier_score'  # Brier score can be > 1 in some cases


def test_config_creation():
    """Test that TrainingConfig is created correctly."""
    config = TrainingConfig(
        model_type='catboost',
        random_seed=123,
        n_trials=50,
        n_splits=3
    )

    assert config.model_type == 'catboost'
    assert config.random_seed == 123
    assert config.n_trials == 50
    assert config.n_splits == 3
    assert config.n_validation_realizations == 1  # Default value


def test_config_validation():
    """Test that TrainingConfig validates inputs correctly."""
    # Valid config should work
    config = TrainingConfig(model_type='xgboost', n_trials=10)
    assert config.model_type == 'xgboost'

    # Invalid model type should raise ValueError
    try:
        TrainingConfig(model_type='invalid_model')
        assert False, "Expected ValueError for invalid model type"
    except ValueError:
        pass  # Expected

    # Invalid n_splits should raise ValueError
    try:
        TrainingConfig(n_splits=1)  # Must be >= 2
        assert False, "Expected ValueError for invalid n_splits"
    except ValueError:
        pass  # Expected

    # Invalid n_trials should raise ValueError
    try:
        TrainingConfig(n_trials=0)  # Must be >= 1
        assert False, "Expected ValueError for invalid n_trials"
    except ValueError:
        pass  # Expected

    # Invalid n_validation_realizations should raise ValueError
    try:
        TrainingConfig(n_validation_realizations=3)  # Must be 1 or 5
        assert False, "Expected ValueError for invalid n_validation_realizations"
    except ValueError:
        pass  # Expected


def test_trainer_initialization():
    """Test that Trainer initializes correctly."""
    config = TrainingConfig()
    trainer = Trainer(config)

    assert trainer.config == config
    assert trainer.model is None
    assert trainer.feature_engineer is None
    assert trainer.best_params is None
    assert trainer.study is None


def test_prepare_data_reshape():
    """Test that _prepare_data correctly reshapes 2D input to 3D."""
    import numpy as np
    import pandas as pd

    # Create a trainer with default config
    config = TrainingConfig()
    trainer = Trainer(config)

    # Create test data: 2D array with 144 features (12 months * 12 bands)
    n_samples = 100
    X_2d = np.random.randn(n_samples, 144)
    y = np.random.randint(0, 2, size=n_samples)

    # Instead of mocking, let's test the actual behavior by checking
    # that we can call _prepare_data without getting a dimension error
    # We'll also verify that the internal feature engineer gets the right shape

    # First, let's make sure the feature engineer is None initially
    assert trainer.feature_engineer is None

    # Call the method under test
    X_features, y_out = trainer._prepare_data(X_2d, y)

    # Verify that we got results back
    assert isinstance(X_features, np.ndarray)
    assert X_features.shape[0] == n_samples  # same number of samples
    assert np.array_equal(y_out, y)  # y should be unchanged

    # Verify that the feature engineer was created
    assert trainer.feature_engineer is not None

    # Also test that 3D input still works
    X_3d = np.random.randn(n_samples, 12, 12)
    trainer2 = Trainer(config)
    X_features2, y_out2 = trainer2._prepare_data(X_3d, y)
    assert isinstance(X_features2, np.ndarray)
    assert X_features2.shape[0] == n_samples
    assert np.array_equal(y_out2, y)

    # Test error cases
    # Wrong number of features in 2D
    try:
        trainer3 = Trainer(config)
        X_wrong = np.random.randn(n_samples, 100)  # Wrong number of features
        trainer3._prepare_data(X_wrong, y)
        assert False, "Should have raised ValueError for wrong feature count"
    except ValueError as e:
        assert "144 features" in str(e)

    # Wrong dimensions
    try:
        trainer4 = Trainer(config)
        X_wrong_dim = np.random.randn(n_samples, 12, 13)  # Wrong last dimension
        trainer4._prepare_data(X_wrong_dim, y)
        assert False, "Should have raised ValueError for wrong dimensions"
    except ValueError as e:
        assert "3-dimensional" in str(e) or "144 features" in str(e)


def test_end_to_end_training_with_2d_input():
    """Test end-to-end training with 2D input (simulating notebook data format)."""
    import numpy as np

    # Create a trainer with minimal settings for fast testing
    config = TrainingConfig(n_trials=2)  # Few trials for quick test
    trainer = Trainer(config)

    # Create test data in the format produced by the notebook:
    # (n_samples, 144) representing 12 months × 12 bands flattened
    n_samples = 50
    X_2d = np.random.randn(n_samples, 144)
    y = np.random.randint(0, 2, size=n_samples)

    # This should work without dimensionality errors
    try:
        trainer.fit(X_2d, y)
        success = True
    except Exception as e:
        success = False
        print(f"Training failed with error: {e}")

    # Verify training succeeded
    assert success, "Training should complete successfully with 2D input"
    assert trainer.model is not None, "Model should be trained"
    assert trainer.feature_engineer is not None, "Feature engineer should be initialized"
    assert hasattr(trainer, 'experiment_dir'), "Experiment directory should be created"


def test_prediction_with_2d_input():
    """Test prediction with 2D input (simulating notebook test data format)."""
    import numpy as np

    # Create a trainer with minimal settings for fast testing
    config = TrainingConfig(n_trials=2)  # Few trials for quick test
    trainer = Trainer(config)

    # Create training data: 2D array with 144 features (12 months * 12 bands)
    n_train = 50
    X_train = np.random.randn(n_train, 144)
    y_train = np.random.randint(0, 2, size=n_train)

    # Create test data: 2D array with 144 features (same format as training)
    n_test = 30
    X_test = np.random.randn(n_test, 144)

    # Train the model
    trainer.fit(X_train, y_train)

    # Test prediction - this should work without dimensionality errors
    try:
        predictions = trainer.predict(X_test)
        prediction_success = True
    except Exception as e:
        prediction_success = False
        print(f"Prediction failed with error: {e}")

    # Test predict_proba - this should also work without dimensionality errors
    try:
        probabilities = trainer.predict_proba(X_test)
        proba_success = True
    except Exception as e:
        proba_success = False
        print(f"Predict_proba failed with error: {e}")

    # Verify both prediction methods worked
    assert prediction_success, "Prediction should complete successfully with 2D input"
    assert proba_success, "Predict_proba should complete successfully with 2D input"
    assert predictions.shape == (n_test,), f"Predictions should have shape ({n_test},)"
    assert probabilities.shape == (n_test, 2), f"Probabilities should have shape ({n_test}, 2)"


if __name__ == "__main__":
    # Run tests if executed directly
    test_model_factory()
    test_competition_score()
    test_calculate_metrics()
    test_config_creation()
    test_config_validation()
    test_trainer_initialization()
    test_prepare_data_reshape()
    test_end_to_end_training_with_2d_input()
    test_prediction_with_2d_input()
    print("All tests passed!")