"""
Input/output utilities for the aquaculture machine learning framework.
"""

from __future__ import annotations

from typing import Union, Dict, Any, Optional, Tuple
import pickle
import json
import yaml
from pathlib import Path
import numpy as np
import pandas as pd

from .config import TrainingConfig
from .model_factory import ModelFactory
from aquaculture.feature_engineering import AquacultureFeatureEngineer


def save_model(model, filepath: Union[str, Path]) -> None:
    """
    Save a trained model to disk using pickle.

    Parameters
    ----------
    model : object
        Trained model to save
    filepath : str or Path
        Path where to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: Union[str, Path]):
    """
    Load a trained model from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved model

    Returns
    -------
    object
        Loaded model
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def save_experiment_config(config: TrainingConfig, filepath: Union[str, Path]) -> None:
    """
    Save experiment configuration to YAML file.

    Parameters
    ----------
    config : TrainingConfig
        Configuration to save
    filepath : str or Path
        Path to save the configuration
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    config.save(filepath)


def load_experiment_config(filepath: Union[str, Path]) -> TrainingConfig:
    """
    Load experiment configuration from YAML file.

    Parameters
    ----------
    filepath : str or Path
        Path to the configuration file

    Returns
    -------
    TrainingConfig
        Loaded configuration
    """
    filepath = Path(filepath)
    return TrainingConfig.load(filepath)


def save_metrics(metrics: dict, filepath: Union[str, Path]) -> None:
    """
    Save metrics dictionary to JSON file.

    Parameters
    ----------
    metrics : dict
        Metrics to save
    filepath : str or Path
        Path to save the metrics
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: Union[str, Path]) -> dict:
    """
    Load metrics from JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to the metrics file

    Returns
    -------
    dict
        Loaded metrics
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def save_predictions(ids: np.ndarray, predictions: np.ndarray,
                    probabilities: np.ndarray, filepath: Union[str, Path]) -> None:
    """
    Save predictions to CSV file for submission.

    Parameters
    ----------
    ids : np.ndarray
        Array of IDs
    predictions : np.ndarray
        Array of predicted classes (0 or 1)
    probabilities : np.ndarray
        Array of predicted probabilities for class 1
    filepath : str or Path
        Path to save the CSV file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame({
        'id': ids,
        'prediction': predictions,
        'probability': probabilities
    })

    # Save to CSV
    df.to_csv(filepath, index=False)


def load_predictions(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load predictions from CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['id', 'prediction', 'probability']
    """
    filepath = Path(filepath)
    return pd.read_csv(filepath)


def create_submission_file(ids: np.ndarray, probabilities: np.ndarray,
                          filepath: Union[str, Path], threshold: float = 0.5) -> None:
    """
    Create a submission file in the required format.

    Parameters
    ----------
    ids : np.ndarray
        Array of IDs
    probabilities : np.ndarray
        Array of predicted probabilities for class 1
    filepath : str or Path
        Path to save the CSV file
    threshold : float, default=0.5
        Threshold for converting probabilities to binary predictions
    """
    # Convert probabilities to binary predictions
    predictions = (probabilities >= threshold).astype(int)

    # Save predictions
    save_predictions(ids, predictions, probabilities, filepath)


def save_feature_importance(feature_names: list, importances: np.ndarray,
                           filepath: Union[str, Path]) -> None:
    """
    Save feature importance to CSV file.

    Parameters
    ----------
    feature_names : list
        List of feature names
    importances : np.ndarray
        Array of feature importance values
    filepath : str or Path
        Path to save the CSV file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame and sort by importance
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    df = df.sort_values('importance', ascending=False)

    # Save to CSV
    df.to_csv(filepath, index=False)


def save_feature_names(feature_names: list, filepath: Union[str, Path]) -> None:
    """
    Save feature names to JSON file.

    Parameters
    ----------
    feature_names : list
        List of feature names
    filepath : str or Path
        Path to save the JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(feature_names, f, indent=2)


def load_feature_names(filepath: Union[str, Path]) -> list:
    """
    Load feature names from JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to the JSON file

    Returns
    -------
    list
        List of feature names
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        feature_names = json.load(f)
    return feature_names


def save_predictions_numpy(predictions: np.ndarray, filepath: Union[str, Path]) -> None:
    """
    Save predictions as numpy array.

    Parameters
    ----------
    predictions : np.ndarray
        Array to save
    filepath : str or Path
        Path to save the file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, predictions)


def load_predictions_numpy(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load predictions from numpy file.

    Parameters
    ----------
    filepath : str or Path
        Path to the file

    Returns
    -------
    np.ndarray
        Loaded array
    """
    filepath = Path(filepath)
    return np.load(filepath)


def save_probabilities(probabilities: np.ndarray, filepath: Union[str, Path]) -> None:
    """
    Save prediction probabilities as numpy array.

    Parameters
    ----------
    probabilities : np.ndarray
        Array to save
    filepath : str or Path
        Path to save the file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(filepath, probabilities)


def load_probabilities(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load prediction probabilities from numpy file.

    Parameters
    ----------
    filepath : str or Path
        Path to the file

    Returns
    -------
    np.ndarray
        Loaded array
    """
    filepath = Path(filepath)
    return np.load(filepath)


def create_experiment_directory(base_dir: Union[str, Path],
                              experiment_name: Optional[str] = None) -> Path:
    """
    Create a timestamped experiment directory.

    Parameters
    ----------
    base_dir : str or Path
        Base directory for experiments
    experiment_name : str, optional
        Optional name to include in the directory name

    Returns
    -------
    pathlib.Path
        Path to the created experiment directory
    """
    from datetime import datetime

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        dir_name = f"{timestamp}_{experiment_name}"
    else:
        dir_name = timestamp

    exp_dir = base_dir / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "features").mkdir(exist_ok=True)
    (exp_dir / "explanations").mkdir(exist_ok=True)
    (exp_dir / "plots").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    return exp_dir


def save_artifacts(trainer, experiment_dir: Union[str, Path]) -> None:
    """
    Save all trainer artifacts to the experiment directory.

    Parameters
    ----------
    trainer : Trainer
        Trained trainer instance
    experiment_dir : str or Path
        Directory to save artifacts to
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save trainer
    trainer_path = experiment_dir / "trainer.pkl"
    with open(trainer_path, 'wb') as f:
        pickle.dump(trainer, f)

    # Save model separately
    if trainer.model is not None:
        model_path = experiment_dir / "models" / "best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(trainer.model, f)

    # Save best parameters
    if trainer.best_params is not None:
        params_path = experiment_dir / "best_params.json"
        with open(params_path, 'w') as f:
            json.dump(trainer.best_params, f, indent=2)

    # Save feature names
    if trainer.feature_names is not None:
        features_path = experiment_dir / "features" / "feature_names.json"
        with open(features_path, 'w') as f:
            json.dump(trainer.feature_names, f, indent=2)

    # Save feature importance if available
    if hasattr(trainer.model, 'feature_importances_') and trainer.feature_names is not None:
        importance_path = experiment_dir / "features" / "feature_importance.csv"
        importance_df = pd.DataFrame({
            'feature': trainer.feature_names,
            'importance': trainer.model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df.to_csv(importance_path, index=False)

    # Save config
    config_path = experiment_dir / "config.yaml"
    trainer.config.save(config_path)

    # Save study if available
    if trainer.study is not None:
        study_path = experiment_dir / "models" / "optuna_study.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(trainer.study, f)


def load_artifacts(experiment_dir: Union[str, Path]) -> tuple:
    """
    Load trainer artifacts from an experiment directory.

    Parameters
    ----------
    experiment_dir : str or Path
        Directory containing the saved artifacts

    Returns
    -------
    tuple
        (trainer, model, best_params, feature_names, study)
    """
    experiment_dir = Path(experiment_dir)

    # Load trainer
    trainer_path = experiment_dir / "trainer.pkl"
    with open(trainer_path, 'rb') as f:
        trainer = pickle.load(f)

    # Load model
    model_path = experiment_dir / "models" / "best_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load best parameters
    params_path = experiment_dir / "best_params.json"
    with open(params_path, 'r') as f:
        best_params = json.load(f)

    # Load feature names
    features_path = experiment_dir / "features" / "feature_names.json"
    with open(features_path, 'r') as f:
        feature_names = json.load(f)

    # Load study
    study_path = experiment_dir / "models" / "optuna_study.pkl"
    try:
        with open(study_path, 'rb') as f:
            study = pickle.load(f)
    except FileNotFoundError:
        study = None

    return trainer, model, best_params, feature_names, study


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
            Predicted probabilities of shape (n_samples,) for probability of class 1
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

    def create_submission(self, ids: np.ndarray, X: np.ndarray,
                         filepath: Union[str, Path], threshold: float = 0.5) -> None:
        """
        Create a submission file from raw input data.

        Parameters
        ----------
        ids : np.ndarray
            Array of IDs
        X : np.ndarray
            Input data of shape (n_samples, 12, 12)
        filepath : str or Path
            Path to save the CSV file
        threshold : float, default=0.5
            Threshold for converting probabilities to binary predictions
        """
        # Get probabilities
        probabilities = self.predict_proba(X)

        # Create submission file
        create_submission_file(ids, probabilities, filepath, threshold)

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
        all_predictions = np.zeros(n_samples, dtype=int)

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = X[i:batch_end]
            batch_predictions = self.predict(batch_X)
            all_predictions[i:batch_end] = batch_predictions

        return all_predictions

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
            Predicted probabilities
        """
        n_samples = X.shape[0]
        all_probabilities = np.zeros(n_samples, dtype=float)

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = X[i:batch_end]
            batch_probabilities = self.predict_proba(batch_X)
            all_probabilities[i:batch_end] = batch_probabilities

        return all_probabilities


def load_inference_pipeline(experiment_dir: Union[str, Path]) -> InferencePipeline:
    """
    Load an inference pipeline from an experiment directory.

    Parameters
    ----------
    experiment_dir : str or Path
        Path to the experiment directory containing the trained model

    Returns
    -------
    InferencePipeline
        Loaded inference pipeline ready for making predictions
    """
    experiment_dir = Path(experiment_dir)

    # Define paths to model components
    model_path = experiment_dir / "models" / "best_model.pkl"
    feature_engineer_path = experiment_dir  # Feature engineer is saved with trainer
    feature_names_path = experiment_dir / "features" / "feature_names.json"

    # Create and return inference pipeline
    return InferencePipeline(
        model_path=model_path,
        feature_engineer_path=feature_engineer_path,
        feature_names_path=feature_names_path
    )


def predict_from_experiment(experiment_dir: Union[str, Path],
                           X: np.ndarray) -> tuple:
    """
    Convenience function to load a model and make predictions in one step.

    Parameters
    ----------
    experiment_dir : str or Path
        Path to the experiment directory
    X : np.ndarray
        Input data of shape (n_samples, 12, 12)

    Returns
    -------
    tuple
        (predictions, probabilities) where predictions are binary labels
        and probabilities are the probability of class 1
    """
    pipeline = load_inference_pipeline(experiment_dir)
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)
    return predictions, probabilities