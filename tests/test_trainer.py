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

if __name__ == "__main__":
    test_predict_and_predict_proba_use_same_features_when_training_true()
    print("Test passed.")