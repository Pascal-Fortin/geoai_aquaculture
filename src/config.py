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
    compute_shap : bool, default=False
        Whether to compute SHAP values for model interpretation
    shap_sample_size : int, default=100
        Number of background samples to use for SHAP explanation (lower values faster but less accurate)
    shap_plot_type : str, default="dot"
        Type of SHAP plot to generate ('dot', 'violin', 'bar')
    shap_max_display : int, default=20
        Maximum number of features to display in SHAP plots
    feature_selection_enabled : bool, default=False
        Whether to enable feature selection
    feature_selection_method : str, default='groups'
        Method to use for feature selection ('groups', 'names', 'patterns', 'indices', 'custom', 'combine')
    feature_selection_kwargs : dict, default={}
        Keyword arguments for the feature selection method
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
    # SHAP configuration options
    compute_shap: bool = False
    shap_sample_size: int = 100
    shap_plot_type: str = "dot"
    shap_max_display: int = 20
    # Feature selection configuration
    feature_selection_enabled: bool = False
    feature_selection_method: str = 'groups'
    feature_selection_kwargs: dict = field(default_factory=dict)

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

        # Validate SHAP parameters
        if self.shap_sample_size < 1:
            raise ValueError("shap_sample_size must be at least 1")

        valid_plot_types = ['dot', 'violin', 'bar']
        if self.shap_plot_type not in valid_plot_types:
            raise ValueError(f"shap_plot_type must be one of {valid_plot_types}")

        if self.shap_max_display < 1:
            raise ValueError("shap_max_display must be at least 1")

        # Initialize feature engineering config if not provided
        if self.feature_engineering_config is None:
            self.feature_engineering_config = AquacultureConfig()

        # Convert experiment_dir to Path object
        self.experiment_dir = Path(self.experiment_dir)

        # Validate feature selection parameters
        if self.feature_selection_enabled:
            valid_methods = ['groups', 'names', 'patterns', 'indices', 'custom', 'combine']
            if self.feature_selection_method not in valid_methods:
                raise ValueError(f"feature_selection_method must be one of {valid_methods}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        # Convert Path objects to strings for YAML serialization
        config_dict['experiment_dir'] = str(self.experiment_dir)
        # Handle AquacultureConfig serialization
        if self.feature_engineering_config:
            feat_dict = asdict(self.feature_engineering_config)
            # Convert tuples to lists for YAML compatibility
            if isinstance(feat_dict.get('window_length_probs'), tuple):
                feat_dict['window_length_probs'] = list(feat_dict['window_length_probs'])
            config_dict['feature_engineering_config'] = feat_dict
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
            # Convert lists back to tuples for tuple fields
            feat_dict = config_dict['feature_engineering_config']
            if 'window_length_probs' in feat_dict and isinstance(feat_dict['window_length_probs'], list):
                feat_dict['window_length_probs'] = tuple(feat_dict['window_length_probs'])
            config_dict['feature_engineering_config'] = AquacultureConfig(**feat_dict)

        # Convert experiment_dir back to Path
        if 'experiment_dir' in config_dict:
            config_dict['experiment_dir'] = Path(config_dict['experiment_dir'])

        return cls(**config_dict)