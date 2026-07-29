"""
Tests for the trainer initialization and configuration saving.
"""

import sys
import os
import tempfile
import tempfile
from pathlib import Path

# Add the project root to the Python path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import TrainingConfig
from src.trainer import Trainer


def test_trainer_initialization_creates_experiment_dir():
    """Test that Trainer initialization creates experiment directory and saves config."""
    # Create a temporary directory for our test
    with tempfile.TemporaryDirectory() as temp_dir:
        base_exp_dir = Path(temp_dir) / "test_experiments"

        # Create a config
        config = TrainingConfig()
        config.experiment_dir = str(base_exp_dir)

        # Create trainer - this should create the experiment directory and save config
        trainer = Trainer(config)

        # Check if experiment directory was created
        assert base_exp_dir.exists(), "Base experiment directory should exist"

        # Check if any experiment subdirectories were created (timestamped)
        exp_subdirs = list(base_exp_dir.iterdir())
        assert len(exp_subdirs) > 0, "Experiment subdirectories should be created"

        # Get the most recent experiment directory (should be the only one)
        latest_exp_dir = exp_subdirs[0]
        assert latest_exp_dir.is_dir(), "Experiment subdirectory should be a directory"

        # Check that config.yaml exists in the experiment directory
        config_path = latest_exp_dir / "config.yaml"
        assert config_path.exists(), "Config file should be created"

        # Check that config file is not empty
        assert config_path.stat().st_size > 0, "Config file should not be empty"

        # Verify the config content can be loaded (using unsafe_load for Python objects)
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.load(f, Loader=yaml.UnsafeLoader)

        # Debug: print what we got
        print(f"Expected experiment_dir: {str(latest_exp_dir)}")
        print(f"Actual experiment_dir in config: {config_data.get('experiment_dir')}")

        # The config should contain the absolute path to the experiment directory
        assert config_data['model_type'] == 'lightgbm'
        # Don't check exact experiment_dir match as it gets converted to absolute path
        # Just check that it's a string and contains our temp directory
        assert isinstance(config_data['experiment_dir'], str)
        assert temp_dir in config_data['experiment_dir']

        print("✓ trainer_initialization_creates_experiment_dir test passed")


def test_trainer_saves_config_during_init():
    """Test that the _save_config method is called during initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_exp_dir = Path(temp_dir) / "test_experiments"

        # Create a custom config
        config = TrainingConfig(
            model_type='catboost',
            random_seed=123,
            n_trials=25,
            n_splits=3,
            learning_rate=0.05
        )
        config.experiment_dir = str(base_exp_dir)

        # Create trainer - this should trigger _create_experiment_directory and _save_config
        trainer = Trainer(config)

        # Check that the experiment directory was created
        assert base_exp_dir.exists(), "Base experiment directory should exist"

        # Check if any experiment subdirectories were created (timestamped)
        exp_subdirs = list(base_exp_dir.iterdir())
        assert len(exp_subdirs) > 0, "Experiment subdirectories should be created"

        # Get the most recent experiment directory (should be the only one)
        latest_exp_dir = exp_subdirs[0]
        assert latest_exp_dir.is_dir(), "Experiment subdirectory should be a directory"

        # Check that config.yaml exists in the experiment directory
        config_path = latest_exp_dir / "config.yaml"
        assert config_path.exists(), "Config file should be created"

        # Verify the config content matches what we set
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.load(f, Loader=yaml.UnsafeLoader)

        assert config_data['model_type'] == 'catboost'
        assert config_data['random_seed'] == 123
        assert config_data['n_trials'] == 25
        assert config_data['n_splits'] == 3
        assert config_data['learning_rate'] == 0.05
        # Check that experiment_dir contains our temp directory
        assert isinstance(config_data['experiment_dir'], str)
        assert temp_dir in config_data['experiment_dir']

        print("✓ trainer_saves_config_during_init test passed")


def test_trainer_inherits_config():
    """Test that trainer properly stores and uses the config."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_exp_dir = Path(temp_dir) / "test_experiments"

        # Create a custom config
        config = TrainingConfig(
            model_type='catboost',
            random_seed=123,
            n_trials=25,
            n_splits=3,
            learning_rate=0.05
        )
        config.experiment_dir = str(base_exp_dir)

        # Create trainer
        trainer = Trainer(config)

        # Check that config is stored
        assert trainer.config == config
        assert trainer.config.model_type == 'catboost'
        assert trainer.config.random_seed == 123
        assert trainer.config.n_trials == 25
        assert trainer.config.n_splits == 3
        assert trainer.config.learning_rate == 0.05

        print("✓ trainer_inherits_config test passed")


if __name__ == "__main__":
    test_trainer_initialization_creates_experiment_dir()
    test_trainer_saves_config_during_init()
    test_trainer_inherits_config()
    print("All Trainer initialization tests passed!")