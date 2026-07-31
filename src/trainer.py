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
from sklearn.model_selection import train_test_split, StratifiedKFold

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
        self._last_X = None
        self._last_X_features = None
        self._last_training = None

        # Set random seeds for reproducibility
        self._set_random_seeds(config.random_seed)

        # Create experiment directory
        self._create_experiment_directory()

        # Save configuration
        self._save_config()

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

    def _setup_file_logging(self, log_dir: Path) -> None:
        """
        Set up file logging to save logs to the experiment directory.

        Parameters
        ----------
        log_dir : pathlib.Path
            Directory where log files should be saved
        """
        # Check if file logging is enabled
        if not self.config.enable_file_logging:
            logger.debug("File logging is disabled in configuration")
            return

        # Create logs directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)

        # Define log file path
        log_file = log_dir / 'training.log'

        # Configure logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Get the log level for file logging (default to INFO)
        try:
            file_log_level = getattr(logging, self.config.log_level_file.upper())
        except AttributeError:
            file_log_level = logging.INFO  # Fallback to INFO if invalid level
            logger.warning(f"Invalid log level '{self.config.log_level_file}', using INFO")

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)

        # Create console handler (to keep existing console output)
        import sys
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Keep console at INFO level
        console_handler.setFormatter(formatter)

        # Get the root logger and add handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Root logger captures all levels

        # Avoid adding duplicate handlers if setup is called multiple times
        if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            root_logger.addHandler(file_handler)
            logger.info(f"File logging enabled: {log_file} (level: {self.config.log_level_file})")

        if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
                   for h in root_logger.handlers):
            root_logger.addHandler(console_handler)

        # DEBUG: Print handlers info
        # print(f"[DEBUG] Root logger handlers: {len(root_logger.handlers)}")
        # for i, h in enumerate(root_logger.handlers):
        #     print(f"  Handler {i}: {type(h).__name__}, level={h.level}")

    def _create_experiment_directory(self) -> Path:
        """
        Create a timestamped experiment directory.

        Returns
        -------
        pathlib.Path
            Path to the created experiment directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(self.config.experiment_dir) / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "features").mkdir(exist_ok=True)
        (exp_dir / "explanations").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        log_dir = exp_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        self.experiment_dir = exp_dir

        # Set up file logging BEFORE doing any logging we want to capture
        self._setup_file_logging(log_dir)

        # Now log the creation (this should go to both console and file)
        logger.info(f"Created experiment directory: {exp_dir}")

        return exp_dir

    def _save_config(self) -> None:
        """Save the training configuration to the experiment directory."""
        config_path = self.experiment_dir / "config.yaml"
        self.config.save(config_path)
        logger.debug(f"Configuration saved to {config_path}")

    def _prepare_data(self, X: np.ndarray, y: np.ndarray, training: bool = False) -> tuple:
        """
        Prepare data by applying feature engineering.

        Parameters
        ----------
        X : np.ndarray
            Raw input data of shape (n_samples, 12, 12) or (n_samples, 144) for flattened features
        y : np.ndarray
            Target labels
        training : bool, default=False
            Whether to apply the observation‑process simulation (windowing + masking) during feature transformation.
            Set to True for training data to enable stochastic window selection and S2‑band dropout.

        Returns
        -------
        tuple
            (X_features, y) where X_features is the engineered feature matrix
        """
        logger.info("Applying feature engineering...")

        # Convert 2D feature matrix to 3D if needed (12 months * 12 bands = 144)
        if X.ndim == 2:
            if X.shape[1] == 144:  # 12 months * 12 bands
                # Reshape from (n_samples, 144) to (n_samples, 12, 12)
                # Assuming column order matches: [VH_01, VV_01, ..., swir2_01, VH_02, ..., swir2_12]
                X = X.reshape((X.shape[0], 12, 12))
            else:
                raise ValueError(
                    f"Expected 2D input with 144 features (12 months × 12 bands), "
                    f"got {X.shape[1]} features"
                )
        elif X.ndim != 3 or X.shape[1] != 12 or X.shape[2] != 12:
            raise ValueError(
                f"Input must be 3-dimensional (n_samples, 12, 12) or 2D with 144 features, "
                f"got shape {X.shape}"
            )

        # Check cache: if same X and training flag as last call, reuse features
        if (self._last_X is not None and self._last_training == training
                and X.shape == self._last_X.shape and np.array_equal(self._last_X, X)):
            logger.debug("Using cached feature matrix.")
            X_features_df = self._last_X_features
        else:
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
                self.feature_engineer.fit(X)

            # Compute features
            X_features_df = self.feature_engineer.transform(X, training=training)

            # Update cache
            self._last_X = X.copy()
            self._last_X_features = X_features_df.copy()
            self._last_training = training

        # Store feature names
        self.feature_names = list(X_features_df.columns)
        logger.info(f"Generated {len(self.feature_names)} features")

        return X_features_df, y

    def _generate_validation_realizations(self, X: np.ndarray, y: np.ndarray) -> list:
        """
        Generate fixed validation realizations for consistent evaluation during Optuna.

        Parameters
        ----------
        X : np.ndarray
            Raw input data of shape (n_samples, 12, 12) or (n_samples, 144) for flattened features
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

            # Prepare data (feature engineering) for this realization
            # We need to temporarily create a feature engineer with the current settings
            # to avoid interfering with the main feature_engineer state
            temp_feature_engineer = AquacultureFeatureEngineer(
                simulate_mask=True,  # Always simulate for validation realizations
                random_state=self.config.random_seed + i * 1000,
                window_length_probs=self.config.feature_engineering_config.window_length_probs,
                start_month_distribution=self.config.feature_engineering_config.start_month_distribution,
                s2_monthly_dropout=self.config.feature_engineering_config.s2_monthly_dropout,
                include_optical=self.config.feature_engineering_config.include_optical,
                include_sar=self.config.feature_engineering_config.include_sar,
                include_cross_sensor_features=self.config.feature_engineering_config.include_cross_sensor_features,
                include_temporal_statistics=self.config.feature_engineering_config.include_temporal_statistics,
                include_metadata=self.config.feature_engineering_config.include_metadata
            )

            # Process the data through the temporary feature engineer
            # This handles both 3D input and 2D input with 144 features
            if X.ndim == 2:
                if X.shape[1] == 144:  # 12 months * 12 bands
                    # Reshape from (n_samples, 144) to (n_samples, 12, 12)
                    # Assuming column order matches: [VH_01, VV_01, ..., swir2_01, VH_02, ..., swir2_12]
                    X_processed = X.reshape((X.shape[0], 12, 12))
                else:
                    raise ValueError(
                        f"Expected 2D input with 144 features (12 months × 12 bands), "
                        f"got {X.shape[1]} features"
                    )
            elif X.ndim == 3 and X.shape[1] == 12 and X.shape[2] == 12:
                # X is already in the correct 3D format
                X_processed = X
            else:
                raise ValueError(
                    f"Input must be 3-dimensional (n_samples, 12, 12) or 2D with 144 features, "
                    f"got shape {X.shape}"
                )

            # Fit and transform the data with the temporary engineer
            temp_feature_engineer.fit(X_processed)
            X_realized = temp_feature_engineer.transform(X_processed, training=True)
            realizations.append((X_realized.values, y))

            logger.debug(f"Generated validation realization {i+1}/{self.config.n_validation_realizations}")

        # Restore original setting
        self.feature_engineer.simulate_mask = original_simulate_mask

        return realizations

    def _generate_validation_realizations_for_fold(self, X: np.ndarray, y: np.ndarray) -> list:
        """
        Generate fixed validation realizations for a specific fold (used during CV).

        This is similar to _generate_validation_realizations but works on fold-specific data.

        Parameters
        ----------
        X : np.ndarray
            Raw input data for the fold of shape (n_samples, 12, 12) or (n_samples, 144) for flattened features
        y : np.ndarray
            Target labels for the fold

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
            self.feature_engineer.random_state = self.config.random_seed + i * 1000 + 10000  # Offset to avoid conflicts

            # Prepare data (feature engineering) for this realization
            # We need to temporarily create a feature engineer with the current settings
            # to avoid interfering with the main feature_engineer state
            temp_feature_engineer = AquacultureFeatureEngineer(
                simulate_mask=True,  # Always simulate for validation realizations
                random_state=self.config.random_seed + i * 1000 + 10000,  # Offset to avoid conflicts
                window_length_probs=self.config.feature_engineering_config.window_length_probs,
                start_month_distribution=self.config.feature_engineering_config.start_month_distribution,
                s2_monthly_dropout=self.config.feature_engineering_config.s2_monthly_dropout,
                include_optical=self.config.feature_engineering_config.include_optical,
                include_sar=self.config.feature_engineering_config.include_sar,
                include_cross_sensor_features=self.config.feature_engineering_config.include_cross_sensor_features,
                include_temporal_statistics=self.config.feature_engineering_config.include_temporal_statistics,
                include_metadata=self.config.feature_engineering_config.include_metadata
            )

            # Process the data through the temporary feature engineer
            # This handles both 3D input and 2D input with 144 features
            if X.ndim == 2:
                if X.shape[1] == 144:  # 12 months * 12 bands
                    # Reshape from (n_samples, 144) to (n_samples, 12, 12)
                    # Assuming column order matches: [VH_01, VV_01, ..., swir2_01, VH_02, ..., swir2_12]
                    X_processed = X.reshape((X.shape[0], 12, 12))
                else:
                    raise ValueError(
                        f"Expected 2D input with 144 features (12 months × 12 bands), "
                        f"got {X.shape[1]} features"
                    )
            elif X.ndim == 3 and X.shape[1] == 12 and X.shape[2] == 12:
                # X is already in the correct 3D format
                X_processed = X
            else:
                raise ValueError(
                    f"Input must be 3-dimensional (n_samples, 12, 12) or 2D with 144 features, "
                    f"got shape {X.shape}"
                )

            # Fit and transform the data with the temporary engineer
            temp_feature_engineer.fit(X_processed)
            X_realized = temp_feature_engineer.transform(X_processed, training=True)
            realizations.append((X_realized.values, y))

            logger.debug(f"Generated validation realization {i+1}/{self.config.n_validation_realizations} for fold")

        # Restore original setting
        self.feature_engineer.simulate_mask = original_simulate_mask

        return realizations

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Trainer':
        """
        Fit the trainer to the training data.

        This method performs the complete training pipeline:
        1. Feature engineering
        2. Train/validation/test split (hold out test set for final evaluation)
        3. Generate validation realizations
        4. Hyperparameter optimization with Optuna
        5. Train final model on combined train+val data (with validation for early stopping)
        6. Evaluate and save artifacts

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, 12, 12) or (n_samples, 144) for flattened features
        y : np.ndarray
            Target labels of shape (n_samples,)

        Returns
        -------
        self : Trainer
            Returns self for method chaining
        """
        # Perform train/validation/test split on RAW data
        # First split: separate out test set (held out for final evaluation)
        # Remaining data is used for cross-validation and hyperparameter tuning

        # Use test_size parameter (validation is handled via cross-validation)
        test_size = getattr(self.config, 'test_size', 0.2)

        # Validate that test_size < 1.0
        if not 0.0 <= test_size < 1.0:
            raise ValueError("test_size must be in [0.0, 1.0)")

        # Single split: separate test set (held out for final evaluation)
        X_train_val_raw, X_test_raw, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.config.random_seed,
            stratify=y
        )

        # The remaining data (X_train_val, y_train_val) will be used for CV and hyperparameter tuning
        # Store classes for later use (from training data for consistency to avoid data leakage)
        self.classes_ = np.unique(y_train_val)

        # Log dataset information
        logger.info(f"Dataset shape: {X.shape}")
        logger.info(f"Training data (for CV): {X_train_val_raw.shape}, Test set (held out): {X_test_raw.shape}")
        logger.info(f"Class distribution (training): {np.bincount(y_train_val)}")

        # Prepare features for TEST dataset (held out for final evaluation)
        X_features_test, _ = self._prepare_data(X_test_raw, y_test, training=True)

        # Set up cross-validation
        n_splits = self.config.n_splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.random_seed)

        # Generate CV splits
        cv_splits = list(skf.split(X_train_val_raw, y_train_val))

        # Pre-compute validation realizations for each fold
        logger.info(f"Generating validation realizations for {n_splits} CV folds...")
        val_features_list = []
        val_labels_list = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            # Get validation data for this fold (raw data)
            X_val_fold_raw = X_train_val_raw[val_idx]
            y_val_fold = y_train_val[val_idx]

            # Generate validation realizations for this fold
            # We use the same logic as _generate_validation_realizations but for fold-specific data
            fold_val_realizations = self._generate_validation_realizations_for_fold(
                X_val_fold_raw, y_val_fold
            )

            # Use the first realization (or average if multiple) for this fold's validation features
            # For consistency with the original approach, we'll use the first realization
            X_val_fold_features, _ = fold_val_realizations[0]

            # Store the preprocessed validation features and labels for this fold
            val_features_list.append(X_val_fold_features)
            val_labels_list.append(y_val_fold)

            logger.debug(f"Fold {fold_idx+1}: Prepared validation features with shape {X_val_fold_features.shape}")

        # Prepare training data (we'll handle feature engineering inside the objective function)
        # For now, we'll just store the raw data - feature engineering happens in objective
        logger.info(f"Starting hyperparameter optimization with {self.config.n_trials} trials using {n_splits}-fold CV...")
        self.study, self.best_params = optimize_hyperparameters(
            X_train=X_train_val_raw,  # Raw training data - feature engineering happens in objective
            y_train=y_train_val,
            val_features_list=val_features_list,  # Precomputed validation features for each fold
            val_labels_list=val_labels_list,      # Validation labels for each fold
            model_type=self.config.model_type,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            random_state=self.config.random_seed,
            n_validation_realizations=self.config.n_validation_realizations,
            n_splits=n_splits,
            feature_engineer_config=self.config.feature_engineering_config,
            study_name=f"aquaculture_{self.config.model_type}_optimization",
            storage=None  # Could be made configurable
        )

        # Save the Optuna study
        study_path = self.experiment_dir / "models" / "optuna_study.pkl"
        from .optuna_utils import save_study
        save_study(self.study, study_path)

        # Train final model with best parameters on ALL training data (everything except held-out test set)
        logger.info("Training final model with best parameters on full training data...")

        # Prepare features for full training data
        X_features_train_full, _ = self._prepare_data(X_train_val_raw, y_train_val, training=True)

        # Train final model (no validation set used, so early stopping is disabled)
        self.model = ModelFactory.create(
            model_type=self.config.model_type,
            y_train=y_train_val,  # Use full training data for class weighting
            **self.best_params
        )

        # Train on full training data
        self.model.fit(X_features_train_full, y_train_val)

        # Evaluate on TEST data (held out from training entirely)
        # We need to prepare features for the test set
        X_features_test, _ = self._prepare_data(X_test_raw, y_test, training=True)

        test_metrics = evaluate_model_performance(
            self.model, X_features_test, y_test, model_name="Test"
        )

        # Save test metrics
        test_metrics_path = self.experiment_dir / "test_metrics.json"
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test metrics saved to {test_metrics_path}")

        # Also evaluate on training data for overfitting comparison
        train_metrics = evaluate_model_performance(
            self.model, X_features_train_full, y_train_val, model_name="Training"
        )

        # For validation information, we report CV statistics from the study
        # Get CV metrics from the best trial
        best_trial = self.study.best_trial
        cv_mean_score = best_trial.user_attrs.get("cv_mean_score", 0.0)
        cv_std_score = best_trial.user_attrs.get("cv_std_score", 0.0)

        # Log evaluation results
        train_score = train_metrics.get('competition_score', 0)
        test_score = test_metrics.get('competition_score', 0)

        logger.info(f"Training score: {train_score:.4f}")
        logger.info(f"Cross Validation (CV) score: {cv_mean_score:.4f} ± {cv_std_score:.4f}")
        logger.info(f"Test set score: {test_score:.4f}")

        # Calculate overfitting indicators (train vs test)
        train_test_gap = train_score - test_score
        logger.info(f"Train-Test gap: {train_test_gap:.4f}")
        if train_test_gap > 0.1:  # Arbitrary threshold for significant overfitting
            logger.warning(f"Large train-test gap ({train_test_gap:.4f}) suggests potential overfitting")

        # Log individual fold scores from the best trial if available
        fold_scores = []
        for i in range(n_splits):
            fold_score = best_trial.user_attrs.get(f"fold_{i}_score", 0.0)
            if fold_score > 0:  # Only include valid scores
                fold_scores.append(fold_score)
        if fold_scores:
            fold_min = min(fold_scores)
            fold_max = max(fold_scores)
            logger.info(f"CV Fold scores: Min={fold_min:.4f}, Max={fold_max:.4f}")

        # Calculate feature importance (using training data statistics)
        if hasattr(self.model, 'feature_importances_') or hasattr(self.model, 'coef_'):
            from .evaluate import get_feature_importance
            importance_df = get_feature_importance(self.model, self.feature_names)

            # Save feature importance
            importance_path = self.experiment_dir / "features" / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")

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

        # Save the trained model separately for inference
        model_path = self.experiment_dir / "models" / "best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Trained model saved to {model_path}")

        logger.info("Training completed successfully!")
        return self

    def predict(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, 12, 12) or (n_samples, 144) for flattened features
        training : bool, default=False
            Whether to apply the observation‑process simulation (windowing + masking) during feature transformation.
            Set to True when evaluating on training data to match the conditions used during fitting.

        Returns
        -------
        np.ndarray
            Predicted class labels
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        # Use _prepare_data which handles reshape/validation and feature engineering.
        # We pass a dummy y array of zeros (length matches X) because _prepare_data expects y,
        # but we only need the features.
        X_features, _ = self._prepare_data(X, np.zeros(X.shape[0]), training=training)
        # X_features is already a numpy array (values) from _prepare_data.

        # Make predictions
        predictions = self.model.predict(X_features)
        return predictions

    def predict_proba(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Predict class probabilities on new data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, 12, 12) or (n_samples, 144) for flattened features
        training : bool, default=False
            Whether to apply the observation‑process simulation (windowing + masking) during feature transformation.
            Set to True when evaluating on training data to match the conditions used during fitting.

        Returns
        -------
        np.ndarray
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        # Use _prepare_data which handles reshape/validation and feature engineering.
        # We pass a dummy y array of zeros (length matches X) because _prepare_data expects y,
        # but we only need the features.
        X_features, _ = self._prepare_data(X, np.zeros(X.shape[0]), training=training)
        # X_features is already a numpy array (values) from _prepare_data.

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