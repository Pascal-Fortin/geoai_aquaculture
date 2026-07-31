"""
Configuration management for the aquaculture machine learning framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# Import the existing aquaculture config
from aquaculture.config import AquacultureConfig


@dataclass
class TrainingConfig:
    """
    Configuration class for the machine learning training pipeline.

    Parameters
    ----------
    model_type : str, default='lightgbm'
        Type of model to use ('lightgbm', 'catboost', or 'xgboost')
    random_seed : int, default=42
        Random seed for reproducibility
    n_splits : int, default=5
        Number of folds for cross-validation
    n_trials : int, default=100
        Number of Optuna optimization trials
    timeout : int, default=3600
        Timeout for Optuna optimization in seconds
    early_stopping_rounds : int, default=50
        Early stopping rounds for model training
    learning_rate : float, default=0.1
        Initial learning rate for models
    n_validation_realizations : int, default=1
        Number of validation realizations to average over (1 or 5)
    experiment_dir : str or Path, default='experiments'
        Base directory for experiment tracking
    feature_engineering_config : AquacultureConfig, optional
        Configuration for the feature engineering pipeline
    test_size : float, default=0.2
        Proportion of dataset to include in the test split (must be between 0.0 and 1.0)
    enable_file_logging : bool, default=True
        Whether to enable file logging to experiment directory
    log_level_file : str, default='INFO'
        Logging level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """

    model_type: str = 'lightgbm'
    random_seed: int = 42
    n_splits: int = 5
    n_trials: int = 100
    timeout: int = 3600
    early_stopping_rounds: int = 50
    learning_rate: float = 0.1
    n_validation_realizations: int = 1
    experiment_dir: Union[str, Path] = 'experiments'
    feature_engineering_config: Optional[AquacultureConfig] = None
    # Data splitting parameters
    test_size: float = 0.2
    # Logging configuration options
    enable_file_logging: bool = True
    log_level_file: str = 'INFO'

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_models = ['lightgbm', 'catboost', 'xgboost']
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}")

        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")

        if not 0.0 <= self.test_size < 1.0:
            raise ValueError("test_size must be in [0.0, 1.0)")

        if self.n_validation_realizations not in [1, 5]:
            raise ValueError("n_validation_realizations must be 1 or 5")

        # Initialize feature engineering config if not provided
        if self.feature_engineering_config is None:
            self.feature_engineering_config = AquacultureConfig()

        # Convert experiment_dir to Path object
        self.experiment_dir = Path(self.experiment_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        # Convert Path objects to strings for YAML serialization
        config_dict['experiment_dir'] = str(self.experiment_dir)
        # Handle AquacultureConfig serialization
        if self.feature_engineering_config:
            config_dict['feature_engineering_config'] = asdict(self.feature_engineering_config)
        return config_dict

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Parameters
        ----------
        filepath : str or Path
            Path to save the configuration file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TrainingConfig':
        """
        Load configuration from YAML file.

        Parameters
        ----------
        filepath : str or Path
            Path to the configuration file

        Returns
        -------
        TrainingConfig
            Loaded configuration instance
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Handle nested AquacultureConfig
        if 'feature_engineering_config' in config_dict and config_dict['feature_engineering_config']:
            from aquaculture.config import AquacultureConfig
            config_dict['feature_engineering_config'] = AquacultureConfig(**config_dict['feature_engineering_config'])

        # Convert experiment_dir back to Path
        if 'experiment_dir' in config_dict:
            config_dict['experiment_dir'] = Path(config_dict['experiment_dir'])

        return cls(**config_dict)