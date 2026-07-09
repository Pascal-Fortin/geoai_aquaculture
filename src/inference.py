"""
Inference module for the aquaculture machine learning framework.
Handles loading trained models and making predictions on new data.
"""

from __future__ import annotations

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json

from .model_factory import ModelFactory
from .io import load_model, load_feature_names


class InferencePipeline:
    """
    Pipeline for making inferences using a trained aquaculture model.

    This class handles loading a trained model and associated preprocessing
    components to make predictions on new data.
    """

    def __init__(self, model_path: Union[str, Path],
                 feature_engineer_path: Optional[Union[str, Path]] = None,
                 feature_names_path: Optional[Union[str, Path]] = None):
        """
        Initialize the inference pipeline.

        Parameters
        ----------
        model_path : str or Path
            Path to the trained model file
        feature_engineer_path : str or Path, optional
            Path to the feature engineering pipeline (AquacultureFeatureEngineer)
            If None, assumes the model includes preprocessing
        feature_names_path : str or Path, optional
            Path to the feature names JSON file
        """
        self.model_path = Path(model_path)
        self.feature_engineer_path = Path(feature_engineer_path) if feature_engineer_path else None
        self.feature_names_path = Path(feature_names_path) if feature_names_path else None

        # Load components
        self.model = self._load_model()
        self.feature_engineer = self._load_feature_engineer()
        self.feature_names = self._load_feature_names()

    def _load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        return load_model(self.model_path)

    def _load_feature_engineer(self):
        """Load the feature engineering pipeline if provided."""
        if self.feature_engineer_path and self.feature_engineer_path.exists():
            from aquaculture.feature_engineering import AquacultureFeatureEngineer
            return AquacultureFeatureEngineer.load(self.feature_engineer_path)
        return None

    def _load_feature_names(self):
        """Load feature names if provided."""
        if self.feature_names_path and self.feature_names_path.exists():
            return load_feature_names(self.feature_names_path)
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make class predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, 12, 12) representing
            [samples, time_steps, bands]

        Returns
        -------
        np.ndarray
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= 0.5).astype(int)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities on new data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, 12, 12) representing
            [samples, time_steps, bands]

        Returns
        -------
        np.ndarray
            Predicted probabilities of shape (n_samples, 2) for [class_0, class_1]
            or (n_samples,) for probability of class 1 if binary classifier
        """
        # Apply feature engineering if available
        if self.feature_engineer is not None:
            X_features = self.feature_engineer.transform(X, training=False)
            X_features = X_features.values
        else:
            # Assume X is already feature-engineered
            X_features = X

        # Make predictions
        probabilities = self.model.predict_proba(X_features)

        # For binary classification, return probability of positive class
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
        else:
            return probabilities

    def predict_batch(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """
        Make predictions in batches to handle large datasets.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, 12, 12)
        batch_size : int, default=1000
            Size of batches for processing

        Returns
        -------
        np.ndarray
            Predicted class labels
        """
        n_samples = X.shape[0]
        predictions = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = X[i:batch_end]
            batch_pred = self.predict(batch_X)
            predictions.append(batch_pred)

        return np.concatenate(predictions)

    def predict_proba_batch(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """
        Predict probabilities in batches to handle large datasets.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, 12, 12)
        batch_size : int, default=1000
            Size of batches for processing

        Returns
        -------
        np.ndarray
            Predicted probabilities for the positive class
        """
        n_samples = X.shape[0]
        probabilities = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = X[i:batch_end]
            batch_prob = self.predict_proba(batch_X)
            probabilities.append(batch_prob)

        return np.concatenate(probabilities)

    def create_submission(self, ids: np.ndarray, X: np.ndarray,
                         output_path: Union[str, Path],
                         threshold: float = 0.5) -> None:
        """
        Create a submission file in the required format.

        Parameters
        ----------
        ids : np.ndarray
            Array of IDs for each sample
        X : np.ndarray
            Input data of shape (n_samples, 12, 12)
        output_path : str or Path
            Path where to save the submission CSV file
        threshold : float, default=0.5
            Threshold for converting probabilities to binary predictions
        """
        # Get probabilities
        probabilities = self.predict_proba(X)

        # Convert to binary predictions
        predictions = (probabilities >= threshold).astype(int)

        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': ids,
            'prediction': predictions,
            'probability': probabilities
        })

        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(output_path, index=False)


def load_inference_pipeline(experiment_dir: Union[str, Path]) -> InferencePipeline:
    """
    Load an inference pipeline from an experiment directory.

    Parameters
    ----------
    experiment_dir : str or Path
        Directory containing the trained model and associated files

    Returns
    -------
    InferencePipeline
        Initialized inference pipeline
    """
    experiment_dir = Path(experiment_dir)

    # Define paths
    model_path = experiment_dir / "models" / "best_model.pkl"
    feature_engineer_path = experiment_dir / "feature_engineer.pkl"
    feature_names_path = experiment_dir / "features" / "feature_names.json"

    # Check if feature engineer exists, otherwise look in aquaculture format
    if not feature_engineer_path.exists():
        # Try alternative location
        feature_engineer_path = experiment_dir / "models" / "feature_engineer.pkl"

    return InferencePipeline(
        model_path=model_path,
        feature_engineer_path=feature_engineer_path if feature_engineer_path.exists() else None,
        feature_names_path=feature_names_path if feature_names_path.exists() else None
    )


def predict_from_experiment(experiment_dir: Union[str, Path],
                           X: np.ndarray,
                           ids: Optional[np.ndarray] = None,
                           output_path: Optional[Union[str, Path]] = None) -> tuple:
    """
    Convenience function to load a model from an experiment directory and make predictions.

    Parameters
    ----------
    experiment_dir : str or Path
        Directory containing the trained model
    X : np.ndarray
        Input data of shape (n_samples, 12, 12)
    ids : np.ndarray, optional
        Array of IDs for each sample (required if output_path is provided)
    output_path : str or Path, optional
        Path to save submission CSV file

    Returns
    -------
    tuple
        (predictions, probabilities) if output_path is None
        None if output_path is provided (saves to file)
    """
    # Load inference pipeline
    pipeline = load_inference_pipeline(experiment_dir)

    # Make predictions
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    # Save submission if requested
    if output_path is not None:
        if ids is None:
            raise ValueError("IDs must be provided when output_path is specified")
        pipeline.create_submission(ids, X, output_path)
        return None
    else:
        return predictions, probabilities