import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.trainer import Trainer
from src.config import TrainingConfig

def test_predict_and_predict_proba_use_same_features_when_training_true():
    """
    Ensure that calling predict and predict_proba with the same input and training=True
    uses the exact same feature matrix (i.e., the stochastic observation process
    is not re-sampled between the two calls).
    """
    # Create a larger deterministic dataset to allow train/val split
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 12, 12))  # 20 samples
    # Create labels with roughly half zeros, half ones
    y = np.array([0]*10 + [1]*10)
    rng.shuffle(y)

    # Use a fixed seed for reproducibility and reduce trials for fast testing
    cfg = TrainingConfig(random_seed=123, n_trials=2)
    trainer = Trainer(cfg)

    # Fit the trainer to initialize the feature engineer
    trainer.fit(X, y)

    # First call: predict
    preds1 = trainer.predict(X, training=True)
    # Second call: predict_proba
    probs1 = trainer.predict_proba(X, training=True)

    # To verify they used the same features, we can compare the internal
    # cached feature matrix from the trainer (if we expose it) or we can
    # compute the features manually via _prepare_data and compare.
    # Simpler: call _prepare_data once and then call model.predict/predict_proba
    # on the returned features; they should match the results from the
    # trainer methods if caching works.
    X_feat, _ = trainer._prepare_data(X, np.zeros(X.shape[0]), training=True)
    preds_direct = trainer.model.predict(X_feat)
    probs_direct = trainer.model.predict_proba(X_feat)

    # The predictions from trainer.predict/trainer.predict_proba should
    # match those from the directly computed features.
    assert np.array_equal(preds1, preds_direct), "predict output differs from direct feature prediction"
    assert np.allclose(probs1, probs_direct), "predict_proba output differs from direct feature probabilities"

    # Additionally, ensure that calling predict then predict_proba (or vice versa)
    # yields the same features by checking that the trainer's cached features
    # are unchanged between calls (we can't access them directly without
    # modifying the class, but we can infer by calling the methods twice and
    # comparing the outputs; if they differ, the cache is not working).
    preds2 = trainer.predict(X, training=True)
    probs2 = trainer.predict_proba(X, training=True)
    assert np.array_equal(preds1, preds2), "predict outputs differ between calls (cache not working)"
    assert np.allclose(probs1, probs2), "predict_proba outputs differ between calls (cache not working)"

    # Finally, ensure that changing the input changes the features (i.e., we are not
    # always returning the same cached values regardless of input).
    X_shift = X.copy()
    X_shift[0, 0, 0] += 50.0  # Very large change to ensure different features
    X_feat_shift, _ = trainer._prepare_data(X_shift, np.zeros(X_shift.shape[0]), training=True)
    assert not np.array_equal(X_feat, X_feat_shift), "Features should change when input changes"


def test_feature_selection_used_in_cv_and_test_data():
    """
    Test that feature selector is used for CV folds and test data when feature selection is enabled.
    """
    # Create a small deterministic dataset for testing
    rng = np.random.default_rng(42)
    n_samples = 30

    # Create data in the expected format: (n_samples, 12, 12)
    X = rng.normal(size=(n_samples, 12, 12))

    # Replace some values with -9999 to simulate missing data
    missing_mask = rng.random(X.shape) < 0.1  # 10% missing
    X[missing_mask] = -9999

    # Create binary labels
    y = rng.integers(0, 2, size=n_samples)

    # Create a training config with feature selection enabled
    config = TrainingConfig(
        random_seed=42,
        n_trials=2,  # Keep small for fast testing
        n_splits=3,  # 3-fold CV
        model_type='lightgbm',
        feature_selection_enabled=True,
        feature_selection_method='groups',
        feature_selection_kwargs={'groups': ['temporal', 'metadata']}  # Select temporal and metadata features
    )

    # Create trainer
    trainer = Trainer(config)

    # Check that feature selector is None before fitting (depends on fitted feature engineer)
    assert trainer.feature_selector is None, "Feature selector should be None before fitting"
    assert trainer.selected_feature_names is None, "Selected feature names should be None before fitting"

    # Fit the trainer (this should use feature selection in CV and for test data)
    trainer.fit(X, y)

    # Check that feature selector was created after fitting
    assert trainer.feature_selector is not None, "Feature selector should be created when enabled"
    assert trainer.selected_feature_names is not None, "Selected feature names should be stored when enabled"

    # Check that we have selected feature names
    assert len(trainer.selected_feature_names) > 0, "Should have selected some features"

    # Verify that the model was trained with the selected features
    if hasattr(trainer.model, 'n_features_in_'):
        assert trainer.model.n_features_in_ == len(trainer.selected_feature_names), \
            f"Model expects {trainer.model.n_features_in_} features but trainer has {len(trainer.selected_feature_names)} selected features"

    # Test prediction to make sure it works
    preds = trainer.predict(X[:5])  # Predict on first 5 samples
    assert preds.shape == (5,), f"Predictions should have shape (5,), got {preds.shape}"

    # Test predict_proba
    probs = trainer.predict_proba(X[:5])  # Predict probabilities on first 5 samples
    assert probs.shape == (5, 2), f"Probabilities should have shape (5, 2), got {probs.shape}"

    # Verify that predictions are valid
    assert np.all(preds >= 0) and np.all(preds <= 1), "Predictions should be between 0 and 1"
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Probabilities should sum to 1 for each sample"


def test_feature_selection_disabled_works_correctly():
    """
    Test that everything works correctly when feature selection is disabled.
    """
    # Create a small deterministic dataset for testing
    rng = np.random.default_rng(42)
    n_samples = 30

    # Create data in the expected format: (n_samples, 12, 12)
    X = rng.normal(size=(n_samples, 12, 12))

    # Replace some values with -9999 to simulate missing data
    missing_mask = rng.random(X.shape) < 0.1  # 10% missing
    X[missing_mask] = -9999

    # Create binary labels
    y = rng.integers(0, 2, size=n_samples)

    # Create a training config with feature selection disabled
    config = TrainingConfig(
        random_seed=42,
        n_trials=2,  # Keep small for fast testing
        n_splits=3,  # 3-fold CV
        model_type='lightgbm',
        feature_selection_enabled=False  # Disabled
    )

    # Create trainer
    trainer = Trainer(config)

    # Check that feature selector was not created
    assert trainer.feature_selector is None, "Feature selector should be None when disabled"
    assert trainer.selected_feature_names is None, "Selected feature names should be None when disabled"

    # Fit the trainer
    trainer.fit(X, y)

    # Check that we have all feature names (no selection)
    assert trainer.feature_names is not None, "Feature names should be stored when selection is disabled"
    assert len(trainer.feature_names) > 0, "Should have feature names"

    # Verify that the model was trained with all features
    if hasattr(trainer.model, 'n_features_in_') and trainer.feature_names is not None:
        assert trainer.model.n_features_in_ == len(trainer.feature_names), \
            f"Model expects {trainer.model.n_features_in_} features but trainer has {len(trainer.feature_names)} features"

    # Test prediction
    preds = trainer.predict(X[:5])  # Predict on first 5 samples
    assert preds.shape == (5,), f"Predictions should have shape (5,), got {preds.shape}"

    # Test predict_proba
    probs = trainer.predict_proba(X[:5])  # Predict probabilities on first 5 samples
    assert probs.shape == (5, 2), f"Probabilities should have shape (5, 2), got {probs.shape}"

    # Verify that predictions are valid
    assert np.all(preds >= 0) and np.all(preds <= 1), "Predictions should be between 0 and 1"
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Probabilities should sum to 1 for each sample"


if __name__ == "__main__":
    test_predict_and_predict_proba_use_same_features_when_training_true()
    test_feature_selection_used_in_cv_and_test_data()
    test_feature_selection_disabled_works_correctly()
    print("All tests passed.")