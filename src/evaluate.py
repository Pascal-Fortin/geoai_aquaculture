"""
Model evaluation utilities for the aquaculture machine learning framework.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import logging
import matplotlib.pyplot as plt
import json
import os

from .metrics import calculate_metrics, competition_score

logger = logging.getLogger(__name__)


def cross_validate_model(model, X: np.ndarray, y: np.ndarray,
                        n_splits: int = 5, random_state: int = 42,
                        calculate_competition_score: bool = True) -> Dict[str, list]:
    """
    Perform stratified cross-validation and return metrics for each fold.

    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate (will be cloned for each fold)
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random seed for reproducible folds
    calculate_competition_score : bool, default=True
        Whether to calculate the competition score (0.6*F1 + 0.4*ROC-AUC)

    Returns
    -------
    dict
        Dictionary containing lists of metric values for each fold:
        - 'competition_score': list of competition scores (if requested)
        - 'f1': list of F1 scores
        - 'roc_auc': list of ROC-AUC scores
        - 'precision': list of precision scores
        - 'recall': list of recall scores
        - 'accuracy': list of accuracy scores
    """
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize results dictionary
    results = {
        'f1': [],
        'roc_auc': [],
        'precision': [],
        'recall': [],
        'accuracy': []
    }
    if calculate_competition_score:
        results['competition_score'] = []

    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.debug(f"Processing fold {fold_idx + 1}/{n_splits}")

        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Clone and fit model
        model_fold = clone(model)
        model_fold.fit(X_train_fold, y_train_fold)

        # Predict probabilities
        y_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]

        # Calculate metrics
        fold_metrics = calculate_metrics(y_val_fold, y_pred_proba)

        # Store results
        for key in results.keys():
            if key in fold_metrics:
                results[key].append(fold_metrics[key])

        logger.debug(f"Fold {fold_idx + 1} - "
                    f"F1: {fold_metrics['f1']:.4f}, "
                    f"ROC-AUC: {fold_metrics['roc_auc']:.4f}, "
                    f"Competition Score: {fold_metrics['competition_score']:.4f}")

    # Log summary statistics
    for key, values in results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

    return results


def evaluate_model_performance(model, X: np.ndarray, y: np.ndarray,
                             model_name: str = "Model") -> Dict[str, float]:
    """
    Evaluate model performance on a dataset.

    Parameters
    ----------
    model : sklearn estimator
        Trained model to evaluate
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    model_name : str, default="Model"
        Name of the model for logging purposes

    Returns
    -------
    dict
        Dictionary of metric values
    """
    # Predict probabilities
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y, y_pred_proba)

    # Log results
    logger.info(f"{model_name} Performance:")
    logger.info(f"  Competition Score: {metrics['competition_score']:.4f}")
    logger.info(f"  F1 Score: {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")

    return metrics


def get_feature_importance(model, feature_names: List[str],
                          importance_type: str = 'auto') -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    Parameters
    ----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute or similar
    feature_names : list of str
        Names of features corresponding to model coefficients
    importance_type : str, default='auto'
        Type of importance to extract ('auto', 'gain', 'split', 'weight')
        For tree-based models, 'gain' is often preferred

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['feature', 'importance'] sorted by importance descending
    """
    # Handle different model types
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (LightGBM, XGBoost, etc.)
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        # CatBoost
        if importance_type == 'auto':
            importance_type = 'Loss'
        importances = model.get_feature_importance(type=importance_type)
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
    else:
        raise AttributeError("Model does not have feature importance attributes")

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # Sort by importance descending
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    return importance_df


def plot_learning_curve(estimator, X: np.ndarray, y: np.ndarray,
                       cv=None, train_sizes=None, scoring=None,
                       title: str = "Learning Curve",
                       ylim: tuple = None,
                       figsize: tuple = (10, 6),
                       save_path: str = None) -> plt.Figure:
    """
    Generate a learning curve showing training and validation scores vs training set size.

    Parameters
    ----------
    estimator : sklearn estimator
        The model to evaluate
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy
    train_sizes : array-like, optional
        Relative or absolute numbers of training examples used for learning curve
    scoring : str, callable, optional
        Scoring metric to use
    title : str, default="Learning Curve"
        Title for the plot
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum y-values plotted
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save the figure. If None, figure is not saved

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    if scoring is None:
        # Use competition score as default scoring
        def competition_scorer(estimator, X, y):
            y_pred_proba = estimator.predict_proba(X)[:, 1]
            return competition_score(y, y_pred_proba)
        scoring = competition_scorer

    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

    # Plot standard deviation bands
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

    ax.grid(True)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curve saved to {save_path}")

    return fig


def plot_validation_curve(estimator, X: np.ndarray, y: np.ndarray, param_name: str,
                         param_range, cv=None, scoring=None,
                         title: str = "Validation Curve",
                         ylim: tuple = None,
                         figsize: tuple = (10, 6),
                         save_path: str = None) -> plt.Figure:
    """
    Generate a validation curve showing training and validation scores vs parameter value.

    Parameters
    ----------
    estimator : sklearn estimator
        The model to evaluate
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    param_name : str
        Name of the parameter to vary
    param_range : array-like
        Values of the parameter to be evaluated
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy
    scoring : str, callable, optional
        Scoring metric to use
    title : str, default="Validation Curve"
        Title for the plot
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum y-values plotted
    figsize : tuple, default=(10, 6)
        Figure size
    save_path : str, optional
        Path to save the figure. If None, figure is not saved

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    if scoring is None:
        # Use competition score as default scoring
        def competition_scorer(estimator, X, y):
            y_pred_proba = estimator.predict_proba(X)[:, 1]
            return competition_score(y, y_pred_proba)
        scoring = competition_scorer

    # Calculate validation curve
    train_scores, test_scores = validation_curve(
        estimator, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(param_range, train_mean, 'o-', color="r", label="Training score")
    ax.plot(param_range, test_mean, 'o-', color="g", label="Validation score")

    # Plot standard deviation bands
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

    ax.grid(True)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Score")
    ax.legend(loc="best")
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Validation curve saved to {save_path}")

    return fig


def plot_training_history(trainer_history: dict,
                         title: str = "Training History",
                         figsize: tuple = (12, 8),
                         save_path: str = None) -> plt.Figure:
    """
    Plot training history showing metrics over epochs/iterations.

    Parameters
    ----------
    trainer_history : dict
        Dictionary containing training history with keys for different metrics
        Expected format: {'train': [metric_values], 'val': [metric_values], ...}
    title : str, default="Training History"
        Title for the plot
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : str, optional
        Path to save the figure. If None, figure is not saved

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    metrics_to_plot = ['competition_score', 'f1', 'roc_auc', 'logloss']
    metric_titles = ['Competition Score', 'F1 Score', 'ROC-AUC', 'Log Loss']

    for idx, (metric, title_text) in enumerate(zip(metrics_to_plot, metric_titles)):
        ax = axes[idx]

        if 'train' in trainer_history and metric in trainer_history['train']:
            epochs = range(1, len(trainer_history['train'][metric]) + 1)
            ax.plot(epochs, trainer_history['train'][metric], 'o-', label=f'Training {title_text}')

        if 'val' in trainer_history and metric in trainer_history['val']:
            epochs = range(1, len(trainer_history['val'][metric]) + 1)
            ax.plot(epochs, trainer_history['val'][metric], 's-', label=f'Validation {title_text}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title_text)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{title_text} over Epochs')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")

    return fig