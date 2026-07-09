"""
Model factory for creating LightGBM, CatBoost, and XGBoost models
with appropriate class weighting for imbalanced datasets.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
from pathlib import Path


class ModelFactory:
    """
    Factory class for creating machine learning models with appropriate
    class weighting for imbalanced datasets.

    Supports LightGBM, CatBoost, and XGBoost classifiers with automatic
    handling of class imbalance through scale_pos_weight or auto_class_weights.
    """

    @staticmethod
    def _calculate_scale_pos_weight(y: np.ndarray) -> float:
        """
        Calculate scale_pos_weight for imbalanced binary classification.

        Parameters
        ----------
        y : np.ndarray
            Binary target array (0s and 1s)

        Returns
        -------
        float
            Scale pos weight value (negative_samples / positive_samples)
        """
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) != 2:
            raise ValueError("Target must be binary for scale_pos_weight calculation")

        n_negative = counts[unique == 0][0] if 0 in unique else 0
        n_positive = counts[unique == 1][0] if 1 in unique else 0

        if n_positive == 0:
            raise ValueError("No positive samples found in target")

        return n_negative / n_positive

    @staticmethod
    def create(model_type: str, random_state: int = 42, **kwargs) -> Union[lgb.LGBMClassifier, cb.CatBoostClassifier, xgb.XGBClassifier]:
        """
        Create a model instance with appropriate class weighting.

        Parameters
        ----------
        model_type : str
            Type of model ('lightgbm', 'catboost', or 'xgboost')
        random_state : int, default=42
            Random seed for reproducibility
        **kwargs : dict
            Additional parameters to pass to the model constructor

        Returns
        -------
        Model instance
            Configured model ready for training
        """
        # Store y for scale_pos_weight calculation if provided
        y_train = kwargs.pop('y_train', None)

        if model_type == 'lightgbm':
            # Handle class weighting for LightGBM
            if y_train is not None and 'scale_pos_weight' not in kwargs:
                scale_pos_weight = ModelFactory._calculate_scale_pos_weight(y_train)
                kwargs['scale_pos_weight'] = scale_pos_weight

            # Set random state
            if 'random_state' not in kwargs:
                kwargs['random_state'] = random_state

            # Set verbosity
            if 'verbose' not in kwargs:
                kwargs['verbose'] = -1

            return lgb.LGBMClassifier(**kwargs)

        elif model_type == 'catboost':
            # Handle class weighting for CatBoost
            if y_train is not None and 'class_weights' not in kwargs:
                # CatBoost uses auto_class_weights or manual class_weights
                kwargs['auto_class_weights'] = 'Balanced'

            # Set random state
            if 'random_seed' not in kwargs:
                kwargs['random_seed'] = random_state

            # Set verbosity
            if 'verbose' not in kwargs:
                kwargs['verbose'] = False

            return cb.CatBoostClassifier(**kwargs)

        elif model_type == 'xgboost':
            # Handle class weighting for XGBoost
            if y_train is not None and 'scale_pos_weight' not in kwargs:
                scale_pos_weight = ModelFactory._calculate_scale_pos_weight(y_train)
                kwargs['scale_pos_weight'] = scale_pos_weight

            # Set random state
            if 'random_state' not in kwargs:
                kwargs['random_state'] = random_state

            # Set verbosity
            if 'verbosity' not in kwargs:
                kwargs['verbosity'] = 0

            return xgb.XGBClassifier(**kwargs)

        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types are: 'lightgbm', 'catboost', 'xgboost'")

    @staticmethod
    def save_model(model: Union[lgb.LGBMClassifier, cb.CatBoostClassifier, xgb.XGBClassifier],
                   filepath: Union[str, Path]) -> None:
        """
        Save a trained model to disk.

        Parameters
        ----------
        model : LightGBM, CatBoost, or XGBoost classifier
            Trained model to save
        filepath : str or Path
            Path where to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)

    @staticmethod
    def load_model(filepath: Union[str, Path]) -> Union[lgb.LGBMClassifier, cb.CatBoostClassifier, xgb.XGBClassifier]:
        """
        Load a trained model from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to the saved model

        Returns
        -------
        Model instance
            Loaded trained model
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)