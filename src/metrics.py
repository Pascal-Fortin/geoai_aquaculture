"""
Metrics calculation utilities for the aquaculture machine learning framework.
"""

from __future__ import annotations

from typing import Dict, Union
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score, accuracy_score,
    precision_recall_curve, roc_curve, brier_score_loss,
    average_precision_score
)


def competition_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate the competition score: 0.6 * F1 + 0.4 * ROC-AUC.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities for the positive class

    Returns
    -------
    float
        Competition score
    """
    # Convert probabilities to binary predictions using threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    # Calculate F1 score
    f1 = f1_score(y_true, y_pred)

    # Calculate ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # This can happen if there's only one class in y_true
        auc = 0.0

    # Calculate competition score
    score = 0.6 * f1 + 0.4 * auc
    return score


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Calculate a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities for the positive class

    Returns
    -------
    dict
        Dictionary containing various metrics:
        - competition_score: 0.6 * F1 + 0.4 * ROC-AUC
        - f1: F1 score
        - roc_auc: ROC-AUC score
        - precision: Precision score
        - recall: Recall score
        - accuracy: Accuracy score
        - brier_score: Brier score (lower is better)
        - pr_auc: Average Precision (Area Under Precision-Recall Curve)
    """
    # Convert probabilities to binary predictions using threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    # Calculate basic metrics
    f1 = f1_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    brier_score = brier_score_loss(y_true, y_prob)

    # Calculate competition score
    comp_score = 0.6 * f1 + 0.4 * roc_auc

    # Calculate average precision (PR-AUC)
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        # This can happen if there's only one class in y_true
        pr_auc = 0.0

    return {
        'competition_score': float(comp_score),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'brier_score': float(brier_score),
        'pr_auc': float(pr_auc)
    }


def calculate_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """
    Calculate precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities for the positive class

    Returns
    -------
    tuple
        (precision, recall, thresholds) arrays
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds


def calculate_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """
    Calculate ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_prob : np.ndarray
        Predicted probabilities for the positive class

    Returns
    -------
    tuple
        (fpr, tpr, thresholds) arrays
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def calculate_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate confusion matrix-based metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels

    Returns
    -------
    dict
        Dictionary containing:
        - tn: True negatives
        - fp: False positives
        - fn: False negatives
        - tp: True positives
        - sensitivity: Same as recall
        - specificity: TN / (TN + FP)
    """
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity)
    }