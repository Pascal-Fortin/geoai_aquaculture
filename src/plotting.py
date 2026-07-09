"""
Plotting utilities for the aquaculture machine learning framework.
"""

from __future__ import annotations

from typing import Union, List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Try to import shap, but don't fail if it's not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. SHAP plots will be disabled.")

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_learning_curve(train_scores: list, val_scores: list,
                       metric_name: str = "Score",
                       save_path: Optional[Union[str, Path]] = None,
                       figsize: tuple = (10, 6)) -> None:
    """
    Plot learning curve showing training and validation scores over epochs/iterations.

    Parameters
    ----------
    train_scores : list
        Training scores per epoch/iteration
    val_scores : list
        Validation scores per epoch/iteration
    metric_name : str, default="Score"
        Name of the metric being plotted
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(10, 6)
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_scores) + 1)
    ax.plot(epochs, train_scores, 'b-', label=f'Training {metric_name}', linewidth=2)
    ax.plot(epochs, val_scores, 'r-', label=f'Validation {metric_name}', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Learning Curve - {metric_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_importance(feature_names: list, importances: np.ndarray,
                           top_n: int = 20,
                           title: str = "Feature Importance",
                           save_path: Optional[Union[str, Path]] = None,
                           figsize: tuple = (10, 8)) -> None:
    """
    Plot feature importance as a horizontal bar chart.

    Parameters
    ----------
    feature_names : list
        List of feature names
    importances : np.ndarray
        Array of feature importance values
    top_n : int, default=20
        Number of top features to display
    title : str, default="Feature Importance"
        Plot title
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(10, 8="Feature Importance"
        Plot title
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(10, 8)
        Figure size
    """
    # Create DataFrame and sort by importance
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    df = df.sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(df))

    ax.barh(y_pos, df['importance'], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()  # Highest values at the top

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float,
                  title: str = "ROC Curve",
                  save_path: Optional[Union[str, Path]] = None,
                  figsize: tuple = (8, 6)) -> None:
    """
    Plot ROC curve.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    auc : float
        Area under the ROC curve
    title : str, default="ROC Curve"
        Plot title
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(8, 6)
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'r--', label='Random classifier', alpha=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray,
                               pr_auc: float,
                               title: str = "Precision-Recall Curve",
                               save_path: Optional[Union[str, Path]] = None,
                               figsize: tuple = (8, 6)) -> None:
    """
    Plot precision-recall curve.

    Parameters
    ----------
    precision : np.ndarray
        Precision values
    recall : np.ndarray
        Recall values
    pr_auc : float
        Area under the precision-recall curve
    title : str, default="Precision-Recall Curve"
        Plot title
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(8, 6)
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, 'b-', label=f'PR curve (AUC = {pr_auc:.3f})', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: list = ['Negative', 'Positive'],
                         title: str = "Confusion Matrix",
                         save_path: Optional[Union[str, Path]] = None,
                         figsize: tuple = (8, 6),
                         normalize: bool = False) -> None:
    """
    Plot confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_names : list, default=['Negative', 'Positive']
        Class names for labeling
    title : str, default="Confusion Matrix"
        Plot title
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(8, 6)
        Figure size
    normalize : bool, default=False
        Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        cmap = 'Blues'
    else:
        fmt = 'd'
        cmap = 'Blues'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Show ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(prob_true: np.ndarray, prob_pred: np.ndarray,
                          title: str = "Calibration Curve",
                          save_path: Optional[Union[str, Path]] = None,
                          figsize: tuple = (8, 6)) -> None:
    """
    Plot calibration curve (reliability diagram).

    Parameters
    ----------
    prob_true : np.ndarray
        True probabilities (fraction of positives)
    prob_pred : np.ndarray
        Predicted probabilities
    title : str, default="Calibration Curve"
        Plot title
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(8, 6)
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "s-", label="Model")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_shap_summary(shap_values: np.ndarray, feature_names: list,
                     X: np.ndarray,
                     plot_type: str = "dot",
                     max_display: int = 20,
                     title: str = "SHAP Summary Plot",
                     save_path: Optional[Union[str, Path]] = None,
                     figsize: tuple = (10, 8)) -> None:
    """
    Plot SHAP summary plot.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    feature_names : list
        List of feature names
    X : np.ndarray
        Feature matrix
    plot_type : str, default="dot"
        Type of plot ('dot', 'violin', 'bar')
    max_display : int, default=20
        Maximum number of features to display
    title : str, default="SHAP Summary Plot"
        Plot title
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(10, 8)
        Figure size
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available. Please install shap to use this function.")

    plt.figure(figsize=figsize)

    if plot_type == "dot":
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                         max_display=max_display, show=False)
    elif plot_type == "violin":
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                         plot_type="violin", max_display=max_display, show=False)
    elif plot_type == "bar":
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                         plot_type="bar", max_display=max_display, show=False)
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}")

    plt.title(title, fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_shap_dependence(shap_values: np.ndarray, feature_names: list,
                        X: np.ndarray, feature_index: int,
                        interaction_index: Optional[int] = None,
                        title: Optional[str] = None,
                        save_path: Optional[Union[str, Path]] = None,
                        figsize: tuple = (10, 8)) -> None:
    """
    Plot SHAP dependence plot for a specific feature.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values
    feature_names : list
        List of feature names
    X : np.ndarray
        Feature matrix
    feature_index : int
        Index of the feature to plot
    interaction_index : int, optional
        Index of the feature for interaction (if None, automatic)
    title : str, optional
        Plot title. If None, uses feature name
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(10, 8)
        Figure size
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not available. Please install shap to use this function.")

    feature_name = feature_names[feature_index]
    if title is None:
        title = f"SHAP Dependence Plot: {feature_name}"

    plt.figure(figsize=figsize)
    shap.dependence_plot(feature_index, shap_values, X,
                        feature_names=feature_names,
                        interaction_index=interaction_index,
                        show=False)
    plt.title(title, fontsize=16, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_correlation_heatmap(X: np.ndarray, feature_names: list,
                                   title: str = "Feature Correlation Heatmap",
                                   save_path: Optional[Union[str, Path]] = None,
                                   figsize: tuple = (12, 10),
                                   method: str = 'pearson') -> None:
    """
    Plot correlation heatmap of features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    feature_names : list
        List of feature names
    title : str, default="Feature Correlation Heatmap"
        Plot title
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(12, 10)
        Figure size
    method : str, default='pearson'
        Correlation method ('pearson', 'spearman', 'kendall')
    """
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)

    # Calculate correlation matrix
    if method == 'pearson':
        corr = df.corr(method='pearson')
    elif method == 'spearman':
        corr = df.corr(method='spearman')
    elif method == 'kendall':
        corr = df.corr(method='kendall')
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Draw heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5},
                ax=ax)

    ax.set_title(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_selection_results(feature_counts: list, scores: list,
                                 metric_name: str = "Score",
                                 title: str = "Feature Selection Results",
                                 save_path: Optional[Union[str, Path]] = None,
                                 figsize: tuple = (10, 6)) -> None:
    """
    Plot feature selection results showing performance vs number of features.

    Parameters
    ----------
    feature_counts : list
        List of numbers of features tested
    scores : list
        List of corresponding scores
    metric_name : str, default="Score"
        Name of the metric being plotted
    title : str, default="Feature Selection Results"
        Plot title
    save_path : str or Path, optional
        Path to save the figure. If None, displays the plot
    figsize : tuple, default=(10, 6)
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(feature_counts, scores, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Features')
    ax.set_ylabel(f'{metric_name}')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Highlight the best point
    best_idx = np.argmax(scores)
    ax.plot(feature_counts[best_idx], scores[best_idx], 'r*', markersize=15,
           label=f'Best: {feature_counts[best_idx]} features ({scores[best_idx]:.4f})')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()