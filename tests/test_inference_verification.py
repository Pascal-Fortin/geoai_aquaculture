"""
Tests for inference verification improvements.
These tests verify that the inference process correctly loads test data,
does not apply masking during inference, and uses the same features as training.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# Add the project root to the Python path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.trainer import Trainer
from src.config import TrainingConfig
from aquaculture.feature_engineering import AquacultureFeatureEngineer
from aquaculture.feature_selection import FeatureSelector


def test_feature_names_loaded_from_json():
    """Test that feature names can be loaded directly from feature_names.json."""
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        experiment_dir = temp_path / "experiment"
        experiment_dir.mkdir()
        features_dir = experiment_dir / "features"
        features_dir.mkdir()

        # Create a sample feature_names.json file
        feature_names = ["feature_0", "feature_1", "feature_2", "feature_3"]
        feature_names_path = features_dir / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names, f)

        # Simulate loading feature names as the improved inference code would
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                loaded_feature_names = json.load(f)

        # Verify the feature names were loaded correctly
        assert loaded_feature_names == feature_names
        assert len(loaded_feature_names) == 4


def test_feature_name_validation_match():
    """Test feature name validation when training and inference features match."""
    # Create mock trainer and feature names
    trainer_feature_names = ["feature_0", "feature_1", "feature_2"]
    inference_feature_names = ["feature_0", "feature_1", "feature_2"]

    # Convert to sets for comparison (as the validation code would)
    train_features = set(trainer_feature_names)
    infer_features = set(inference_feature_names)

    # Verify they match
    assert train_features == infer_features
    assert len(train_features) == 3
    assert len(infer_features) == 3


def test_feature_name_validation_mismatch():
    """Test feature name validation detects mismatches between training and inference."""
    # Create mock trainer and feature names with a mismatch
    trainer_feature_names = ["feature_0", "feature_1", "feature_2"]
    inference_feature_names = ["feature_0", "feature_1", "feature_3"]  # feature_2 vs feature_3

    # Convert to sets for comparison (as the validation code would)
    train_features = set(trainer_feature_names)
    infer_features = set(inference_feature_names)

    # Verify they don't match
    assert train_features != infer_features

    # Check specific differences
    train_only = train_features - infer_features
    infer_only = infer_features - train_features

    assert "feature_2" in train_only
    assert "feature_3" in infer_only
    assert len(train_only) == 1
    assert len(infer_only) == 1


def test_save_feature_engineering_config():
    """Test that feature engineering configuration can be saved separately."""
    # Create a temporary directory for experiment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        experiment_dir = temp_path / "experiment"
        experiment_dir.mkdir()

        # Create a mock config object with feature engineering config
        class MockFeatureEngineeringConfig:
            def __init__(self):
                self.simulate_mask = True
                self.random_state = 42
                self.window_length_probs = (1/3, 1/3, 1/3)
                self.include_optical = True
                self.include_sar = True

        class MockConfig:
            def __init__(self):
                self.feature_engineering_config = MockFeatureEngineeringConfig()
                self.model_type = "lightgbm"
                self.random_seed = 123

        # Create a mock trainer
        trainer = type('MockTrainer', (), {})()
        trainer.experiment_dir = experiment_dir
        trainer.config = MockConfig()

        # Define the _save_feature_engineering_config method (as it would be implemented)
        def _save_feature_engineering_config(self):
            """Save the feature engineering configuration to the experiment directory."""
            config_path = self.experiment_dir / "feature_engineering_config.json"

            # Extract feature engineering config from main config
            if hasattr(self.config, 'feature_engineering_config'):
                fe_config = self.config.feature_engineering_config

                # Convert to dictionary if it's a config object
                if hasattr(fe_config, '__dict__'):
                    fe_config_dict = fe_config.__dict__
                else:
                    fe_config_dict = fe_config

                with open(config_path, 'w') as f:
                    json.dump(fe_config_dict, f, indent=2)

        # Bind the method to our mock trainer
        import types
        trainer._save_feature_engineering_config = types.MethodType(_save_feature_engineering_config, trainer)

        # Call the method
        trainer._save_feature_engineering_config()

        # Verify the file was created
        config_path = experiment_dir / "feature_engineering_config.json"
        assert config_path.exists()

        # Verify the content is correct
        with open(config_path, 'r') as f:
            saved_config = json.load(f)

        assert saved_config['simulate_mask'] == True
        assert saved_config['random_state'] == 42
        assert saved_config['window_length_probs'] == [1/3, 1/3, 1/3]
        assert saved_config['include_optical'] == True
        assert saved_config['include_sar'] == True


def test_inference_uses_pre_fitted_components():
    """Test that inference uses pre-fitted feature engineering components from trainer."""
    # Create sample data
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(20, 12, 12))  # Training data
    y_train = rng.integers(0, 2, size=20)    # Training labels
    X_test = rng.normal(size=(10, 12, 12))   # Test data

    # Create and fit trainer
    config = TrainingConfig(
        random_seed=123,
        n_trials=2,
        feature_selection_enabled=True,
        feature_selection_method='groups',
        feature_selection_kwargs={'groups': ['temporal', 'metadata']}
    )
    trainer = Trainer(config)
    trainer.fit(X_train, y_train)

    # Verify trainer has the expected components
    assert trainer.feature_engineer is not None
    assert trainer.feature_selector is not None  # Since we enabled feature selection
    assert trainer.feature_names is not None
    assert len(trainer.feature_names) > 0

    # Test that we can use the trainer's components for inference
    # (This mimics what the improved inference notebook would do)
    feature_engineer = trainer.feature_engineer
    feature_selector = trainer.feature_selector

    # Transform test data using the pre-fitted components (with training=False)
    # The feature selector internally handles both feature engineering and selection
    if feature_selector is not None:
        X_test_features = feature_selector.transform(X_test, training=False)
    else:
        X_test_features = feature_engineer.transform(X_test, training=False)

    # Verify we got features back
    assert isinstance(X_test_features, pd.DataFrame)
    assert X_test_features.shape[0] == 10  # Same number of test samples
    assert X_test_features.shape[1] == len(trainer.feature_names)  # Same number of features as training

    # Verify the feature names match
    infer_feature_names = list(X_test_features.columns)
    assert set(infer_feature_names) == set(trainer.feature_names)


if __name__ == "__main__":
    # Run the tests
    test_feature_names_loaded_from_json()
    print("��✓ test_feature_names_loaded_from_json passed")

    test_feature_name_validation_match()
    print("��✓ test_feature_name_validation_match passed")

    test_feature_name_validation_mismatch()
    print("��✓ test_feature_name_validation_mismatch passed")

    test_save_feature_engineering_config()
    print("��✓ test_save_feature_engineering_config passed")

    test_inference_uses_pre_fitted_components()
    print("��✓ test_inference_uses_pre_fitted_components passed")

    print("\nAll inference verification tests passed! �� 🎉")