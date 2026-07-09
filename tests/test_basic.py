"""
Basic tests for the aquaculture machine learning framework.
"""

import numpy as np
import pytest
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
    with pytest.raises(ValueError):
        TrainingConfig(model_type='invalid_model')

    # Invalid n_splits should raise ValueError
    with pytest.raises(ValueError):
        TrainingConfig(n_splots=1)  # Must be >= 2

    # Invalid n_trials should raise ValueError
    with pytest.raises(ValueError):
        TrainingConfig(n_trials=0)  # Must be >= 1

    # Invalid n_validation_realizations should raise ValueError
    with pytest.raises(ValueError):
        TrainingConfig(n_validation_realizations=3)  # Must be 1 or 5


def test_trainer_initialization():
    """Test that Trainer initializes correctly."""
    config = TrainingConfig()
    trainer = Trainer(config)

    assert trainer.config == config
    assert trainer.model is None
    assert trainer.feature_engineer is None
    assert trainer.best_params is None
    assert trainer.study is None


if __name__ == "__main__":
    # Run tests if executed directly
    test_model_factory()
    test_competition_score()
    test_calculate_metrics()
    test_config_creation()
    test_config_validation()
    test_trainer_initialization()
    print("All tests passed!")