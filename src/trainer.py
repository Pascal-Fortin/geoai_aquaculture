"""
Trainer class for the aquaculture machine learning framework.
Orchestrates feature engineering, observation process simulation,
cross-validation, hyperparameter optimization, and model training.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings

from .config import TrainingConfig
from .model_factory import ModelFactory
from .optuna_utils import create_objective_function, create_optuna_study, optimize_hyperparameters
from .evaluate import cross_validate_model, evaluate_model_performance
from .metrics import competition_score, calculate_metrics
from aquaculture.feature_engineering import AquacultureFeatureEngineer

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main trainer class for the aquaculture machine learning pipeline.

    This class orchestrates the entire machine learning workflow:
    1. Feature engineering using AquacultureFeatureEngineer
    2. Observation process simulation (stochastic for training, fixed for validation)
    3. Cross-validation with StratifiedKFold
    4. Hyperparameter optimization using Optuna
    5. Model training with best parameters
    6. Evaluation and artifact saving

    Parameters
    ----------
    config : TrainingConfig
        Configuration object containing all training parameters
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the Trainer.

        Parameters
        ----------
        config : TrainingConfig
            Training configuration
        """
        self.config = config
        self.feature_engineer = None
        self.model = None
        self.best_params = None
        self.study = None
        self.experiment_dir = None
        self.feature_names = None
        self.classes_ = None

        # Set random seeds for reproducibility
        self._set_random_seeds(config.random_seed)

        logger.info(f"Trainer initialized with model_type={config.model_type}")

    def _set_random_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random
        import os

        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        # Try to set seeds for various libraries
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass  # PyTorch not installed

    def _create_experiment_directory(self) -> Path:
        """
        Create a timestamped experiment directory.

        Returns
        -------
        pathlib.Path
            Path to the created experiment directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.config.experiment_dir / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "features").mkdir(exist_ok=True)
        (exp_dir / "explanations").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)

        self.experiment_dir = exp_dir
        logger.info(f"Created experiment directory: {exp_dir}")

        return exp_dir

    def _save_config(self) -> None:
        """Save the training configuration to the experiment directory."""
        config_path = self.experiment_dir / "config.yaml"
        self.config.save(config_path)
        logger.debug(f"Configuration saved to {config_path}")

    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Prepare data by applying feature engineering.

        Parameters
        ----------
        X : np.ndarray
            Raw input data of shape (n_samples, 12, 12)
        y : np.ndarray
            Target labels

        Returns
        -------
        tuple
            (X_features, y) where X_features is the engineered feature matrix
        """
        logger.info("Applying feature engineering...")

        # Initialize feature engineer if not already done
        if self.feature_engineer is None:
            self.feature_engineer = AquacultureFeatureEngineer(
                simulate_mask=self.config.feature_engineering_config.simulate_mask,
                random_state=self.config.random_seed,
                window_length_probs=self.config.feature_engineering_config.window_length_probs,
                start_month_distribution=self.config.feature_engineering_config.start_month_distribution,
                s2_monthly_dropout=self.config.feature_engineering_config.s2_monthly_dropout,
                include_optical=self.config.feature_engineering_config.include_optical,
                include_sar=self.config.feature_engineering_config.include_sar,
                include_cross_sensor_features=self.config.feature_engineering_config.include_cross_sensor_features,
                include_temporal_statistics=self.config.feature_engineering_config.include_temporal_statistics,
                include_metadata=self.config.feature_engineering_config.include_metadata
            )

        # Fit and transform the data
        self.feature_engineer.fit(X)
        X_features = self.feature_engineer.transform(X, training=False)  # We'll handle observation process separately

        # Store feature names
        self.feature_names = list(X_features.columns)
        logger.info(f"Generated {len(self.feature_names)} features")

        return X_features.values, y

    def _generate_validation_realizations(self, X: np.ndarray, y: np.ndarray) -> list:
        """
        Generate fixed validation realizations for consistent evaluation during Optuna.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels

        Returns
        -------
        list
            List of tuples (X_realized, y) for each validation realization
        """
        realizations = []

        # Store original simulate_mask setting
        original_simulate_mask = self.feature_engineer.simulate_mask

        for i in range(self.config.n_validation_realizations):
            # Enable simulation for validation realization generation
            self.feature_engineer.simulate_mask = True
            # Use different random seed for each realization
            self.feature_engineer.random_state = self.config.random_seed + i * 1000

            # Transform data with observation process simulation
            X_realized = self.feature_engineer.transform(X, training=False)
            realizations.append((X_realized.values, y))

            logger.debug(f"Generated validation realization {i+1}/{self.config.n_validation_realizations}")

        # Restore original setting
        self.feature_engineer.simulate_mask = original_simulate_mask

        return realizations

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Trainer':
        """
        Fit the trainer to the training data.

        This method performs the complete training pipeline:
        1. Feature engineering
        2. Train/validation split
        3. Generate validation realizations
        4. Hyperparameter optimization with Optuna
        5. Train final model on full dataset
        6. Evaluate and save artifacts

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, 12, 12)
        y : np.ndarray
            Target labels of shape (n_samples,)

        Returns
        -------
        self : Trainer
            Returns self for method chaining
        """
        # Create experiment directory
        self._create_experiment_directory()

        # Save configuration
        self._save_config()

        # Prepare data (feature engineering)
        X_features, y = self._prepare_data(X, y)

        # Store classes for later use
        self.classes_ = np.unique(y)

        # Log dataset information
        logger.info(f"Dataset shape: {X_features.shape}")
        logger.info(f"Class distribution: {np.bincount(y)}")

        # Perform train/validation split for hyperparameter tuning
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_features, y,
            test_size=0.2,
            random_state=self.config.random_seed,
            stratify=y
        )

        logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")

        # Generate fixed validation realizations
        logger.info(f"Generating {self.config.n_validation_realizations} validation realization(s)...")
        val_realizations = self._generate_validation_realizations(X_val, y_val)

        # Optimize hyperparameters using Optuna
        logger.info(f"Starting hyperparameter optimization with {self.config.n_trials} trials...")
        self.study, self.best_params = optimize_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,  # We'll use the first realization for optimization
            y_val=y_val,
            model_type=self.config.model_type,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            random_state=self.config.random_seed,
            n_validation_realizations=self.config.n_validation_realizations,
            study_name=f"aquaculture_{self.config.model_type}_optimization",
            storage=None  # Could be made configurable
        )

        # Save the Optuna study
        study_path = self.experiment_dir / "models" / "optuna_study.pkl"
        from .optuna_utils import save_study
        save_study(self.study, study_path)

        # Train final model with best parameters on full training data
        logger.info("Training final model with best parameters...")
        self.model = ModelFactory.create(
            model_type=self.config.model_type,
            random_state=self.config.random_seed,
            y_train=y,  # Use full dataset for class weighting
            **self.best_params
        )

        # Fit on full dataset
        self.model.fit(X_features, y)

        # Evaluate on training data
        train_metrics = evaluate_model_performance(
            self.model, X_features, y, model_name="Training"
        )

        # Save training metrics
        train_metrics_path = self.experiment_dir / "metrics.json"
        with open(train_metrics_path, 'w') as f:
            json.dump(train_metrics, f, indent=2)

        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_') or hasattr(self.model, 'coef_'):
            from .evaluate import get_feature_importance
            importance_df = get_feature_importance(self.model, self.feature_names)

            # Save feature importance
            importance_path = self.experiment_dir / "features" / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")

        # Save the trained model
        model_path = self.experiment_dir / "models" / "best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {model_path}")

        # Save feature names
        feature_names_path = self.experiment_dir / "features" / "feature_names.json"
        with open(feature_names_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        logger.info(f"Feature names saved to {feature_names_path}")

        # Save best parameters
        params_path = self.experiment_dir / "best_params.json"
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        logger.info(f"Best parameters saved to {params_path}")

        logger.info("Training completed successfully!")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, 12, 12)

        Returns
        -------
        np.ndarray
            Predicted class labels
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        # Apply feature engineering
        X_features = self.feature_engineer.transform(X, training=False)
        X_features = X_features.values

        # Make predictions
        predictions = self.model.predict(X_features)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities on new data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, 12, 12)

        Returns
        -------
        np.ndarray
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        # Apply feature engineering
        X_features = self.feature_engineer.transform(X, training=False)
        X_features = X_features.values

        # Predict probabilities
        probabilities = self.model.predict_proba(X_features)
        return probabilities

    def save(self, directory: Optional[Union[str, Path]] = None) -> None:
        """
        Save the trainer and all associated artifacts.

        Parameters
        ----------
        directory : str or Path, optional
            Directory to save to. If None, uses the experiment directory
        """
        if directory is None:
            directory = self.experiment_dir
        else:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)

        # Save the trainer itself
        trainer_path = directory / "trainer.pkl"
        with open(trainer_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Trainer saved to {trainer_path}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Trainer':
        """
        Load a trainer from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved trainer

        Returns
        -------
        Trainer
            Loaded trainer instance
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            trainer = pickle.load(f)
        logger.info(f"Trainer loaded from {filepath}")
        return trainer