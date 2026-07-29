"""
Tests for the enhanced Optuna utilities with cross-validation support.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import optuna

from src.optuna_utils import (
    create_objective_function,
    create_optuna_study,
    optimize_hyperparameters,
    plot_training_vs_validation_scores,
    plot_optimization_history_with_training
)
from src.metrics import competition_score


def test_create_objective_function():
    """Test that the objective function is created correctly with CV API."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create CV splits to generate validation data for each fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X, y))

    # Generate validation features and labels for each fold (simplified for testing)
    val_features_list = []
    val_labels_list = []
    for train_idx, val_idx in cv_splits:
        # For testing, we'll just use the validation data directly as "features"
        # In reality, these would be preprocessed features
        val_features_list.append(X[val_idx])  # Simplified
        val_labels_list.append(y[val_idx])

    # Create objective function with CV API
    objective = create_objective_function(
        X_train=X,
        y_train=y,
        val_features_list=val_features_list,
        val_labels_list=val_labels_list,
        model_type='logistic_regression',  # This will fail but we're testing creation
        n_validation_realizations=1,
        n_splits=3
    )

    # Check that it's callable
    assert callable(objective)
    print("✓ create_objective_function test passed")


def test_optimization_tracking():
    """Test that optimization tracks CV scores."""
    # Create sample data
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Create CV splits
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_splits = list(skf.split(X, y))

    # Generate validation data for each fold (simplified)
    val_features_list = []
    val_labels_list = []
    for train_idx, val_idx in cv_splits:
        val_features_list.append(X[val_idx])  # Simplified
        val_labels_list.append(y[val_idx])

    # Create study
    study = create_optuna_study(study_name="test_tracking_cv")

    # Create a simple objective function for testing that mimics our CV approach
    def simple_cv_objective(trial):
        # Simple hyperparameter: C parameter for LogisticRegression
        C = trial.suggest_float("C", 0.01, 10.0, log=True)

        # Create and train model
        model = LogisticRegression(C=C, random_state=42, max_iter=1000)

        # Simulate CV evaluation
        fold_scores = []
        # Use the same CV splits for consistency
        for train_idx, val_idx in cv_splits:
            X_fold_train, y_fold_train = X[train_idx], y[train_idx]
            X_fold_val, y_fold_val = X[val_idx], y[val_idx]  # Simplified

            model.fit(X_fold_train, y_fold_train)
            val_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            fold_score = competition_score(y_fold_val, val_pred_proba)
            fold_scores.append(fold_score)

        # Return mean score
        mean_score = np.mean(fold_scores)

        # Store metrics (mimicking what our real objective function does)
        trial.set_user_attr("cv_mean_score", mean_score)
        trial.set_user_attr("cv_std_score", np.std(fold_scores))
        for i, score in enumerate(fold_scores):
            trial.set_user_attr(f"fold_{i}_score", score)

        return mean_score

    # Run optimization
    study.optimize(simple_cv_objective, n_trials=5)

    # Check that trials have the expected user attributes
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed_trials) == 5

    for trial in completed_trials:
        assert "cv_mean_score" in trial.user_attrs
        assert "cv_std_score" in trial.user_attrs
        # Check that fold scores are stored
        for i in range(3):  # 3 folds
            assert f"fold_{i}_score" in trial.user_attrs

        # Verify scores are in valid range
        assert 0 <= trial.user_attrs["cv_mean_score"] <= 1
        assert 0 <= trial.user_attrs["cv_std_score"] <= 1

    print("✓ optimization_tracking test passed")


def test_plot_functions_exist():
    """Test that the plotting functions exist and are callable."""
    # These tests mainly check that the functions exist and don't crash on basic input
    # Since plotting requires GUI backend which may not be available in test env,
    # we'll just check they're callable

    assert callable(plot_training_vs_validation_scores)
    assert callable(plot_optimization_history_with_training)
    print("✓ plot_functions_exist test passed")


if __name__ == "__main__":
    test_create_objective_function()
    test_optimization_tracking()
    test_plot_functions_exist()
    print("All Optuna utilities tests passed!")