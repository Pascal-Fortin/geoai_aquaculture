"""
Optuna utilities for hyperparameter optimization in the aquaculture machine learning framework.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Callable, Tuple, Sequence, List
import optuna
import numpy as np
import pandas as pd
import math
import logging
from pathlib import Path
import joblib
from sklearn.model_selection import StratifiedKFold

from .model_factory import ModelFactory
from .metrics import competition_score, calculate_metrics
from .evaluate import cross_validate_model
from aquaculture.feature_engineering import AquacultureFeatureEngineer

logger = logging.getLogger(__name__)


def create_optuna_study(study_name: str, storage: Optional[str] = None,
                       load_if_exists: bool = True,
                       sampler: Optional[optuna.samplers.BaseSampler] = None,
                       pruner: Optional[optuna.pruners.BasePruner] = None) -> optuna.Study:
    """
    Create an Optuna study for hyperparameter optimization.

    Parameters
    ----------
    study_name : str
        Name of the study
    storage : str, optional
        Database URL for persistent storage (e.g., 'sqlite:///example.db')
    load_if_exists : bool, default=True
        Whether to load existing study if it exists
    sampler : optuna.samplers.BaseSampler, optional
        Sampler algorithm (default: TPESampler)
    pruner : optuna.pruners.BasePruner, optional
        Pruner algorithm (default: MedianPruner)

    Returns
    -------
    optuna.Study
        Created Optuna study
    """
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=42)

    if pruner is None:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction='maximize',  # We want to maximize the competition score
        sampler=sampler,
        pruner=pruner
    )

    logger.info(f"Created Optuna study '{study_name}'")
    if storage:
        logger.info(f"Using storage: {storage}")

    return study


def create_objective_function(X_train: np.ndarray, y_train: np.ndarray,
                            val_features_list: list, val_labels_list: list,
                            model_type: str, random_state: int = 42,
                            n_validation_realizations: int = 1,
                            n_splits: int = 5,
                            feature_engineer_config = None) -> Callable[[optuna.Trial], float]:
    """
    Create an objective function for Optuna optimization with Stratified K-Fold CV.

    Parameters
    ----------
    X_train : np.ndarray
        Full training features
    y_train : np.ndarray
        Full training targets
    val_features_list : list
        List of precomputed validation features for each fold
    val_labels_list : list
        List of validation labels for each fold
    model_type : str
        Type of model ('lightgbm', 'catboost', or 'xgboost')
    random_state : int, default=42
        Random seed for reproducibility
    n_validation_realizations : int, default=1
        Number of validation realizations to average over
    n_splits : int, default=5
        Number of CV folds
    feature_engineer_config : aquaculture.config.AquacultureConfig, optional
        Configuration for the feature engineering pipeline

    Returns
    -------
    callable
        Objective function that takes an Optuna trial and returns a score to maximize
    """
    def objective(trial: optuna.Trial) -> float:
        # Define hyperparameter search space based on model type
        if model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }

        elif model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 10),
            }

        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
            }

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Add random state
        if model_type == 'lightgbm':
            params['random_state'] = random_state
        elif model_type == 'catboost':
            params['random_seed'] = random_state
        elif model_type == 'xgboost':
            params['random_state'] = random_state

        try:
            # Create model with suggested parameters
            model = ModelFactory.create(
                model_type=model_type,
                y_train=y_train,  # For class weighting
                **params
            )

            # Perform Stratified K-Fold cross-validation
            fold_scores = []

            # For each fold, we need to get the training/validation split
            # We'll create the splits inside the objective function to ensure
            # that each trial gets different realizations for training data
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                # Get training and validation data for this fold
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                # Use pre-computed validation features for this fold
                X_fold_val = val_features_list[fold_idx]  # Already processed features
                y_fold_val = val_labels_list[fold_idx]

                # Handle feature_engineer_config - provide default if not passed
                if feature_engineer_config is None:
                    # Import here to avoid circular imports
                    from aquaculture.config import AquacultureConfig
                    fec = AquacultureConfig()
                else:
                    fec = feature_engineer_config

                # Apply feature engineering to training data for this trial
                # Use a different seed for each trial to ensure different realizations
                feature_engineer_for_trial = AquacultureFeatureEngineer(
                    simulate_mask=fec.simulate_mask,
                    random_state=random_state + trial.number * 1000,  # Different seed per trial
                    window_length_probs=fec.window_length_probs,
                    start_month_distribution=fec.start_month_distribution,
                    s2_monthly_dropout=fec.s2_monthly_dropout,
                    include_optical=fec.include_optical,
                    include_sar=fec.include_sar,
                    include_cross_sensor_features=fec.include_cross_sensor_features,
                    include_temporal_statistics=fec.include_temporal_statistics,
                    include_metadata=fec.include_metadata
                )

                # Process training data (3D raw -> 2D features)
                if X_fold_train.ndim == 2:
                    if X_fold_train.shape[1] == 144:  # 12 months * 12 bands
                        X_fold_train_processed = X_fold_train.reshape((X_fold_train.shape[0], 12, 12))
                    else:
                        raise ValueError(
                            f"Expected 2D input with 144 features (12 months × 12 bands), "
                            f"got {X_fold_train.shape[1]} features"
                        )
                elif X_fold_train.ndim == 3 and X_fold_train.shape[1] == 12 and X_fold_train.shape[2] == 12:
                    # X is already in the correct 3D format
                    X_fold_train_processed = X_fold_train
                else:
                    raise ValueError(
                        f"Input must be 3-dimensional (n_samples, 12, 12) or 2D with 144 features, "
                        f"got shape {X_fold_train.shape}"
                    )

                feature_engineer_for_trial.fit(X_fold_train_processed)
                X_fold_train_features = feature_engineer_for_trial.transform(X_fold_train_processed, training=True).values

                # Validation features are already precomputed (2D)
                X_fold_val_features = X_fold_val  # Already processed

                # Train model
                model.fit(X_fold_train_features, y_fold_train)

                # Predict and evaluate
                y_pred_proba = model.predict_proba(X_fold_val_features)[:, 1]
                fold_score = competition_score(y_fold_val, y_pred_proba)
                fold_scores.append(fold_score)

            # Calculate statistics across folds
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            # Store metrics in trial
            trial.set_user_attr("cv_mean_score", mean_score)
            trial.set_user_attr("cv_std_score", std_score)
            for i, score in enumerate(fold_scores):
                trial.set_user_attr(f"fold_{i}_score", score)

            # Return mean score for optimization
            return mean_score

        except Exception as e:
            logger.warning(f"Trial failed with error: {e}")
            # Return a very poor score to discourage this parameter setting
            return 0.0

    return objective


def optimize_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                           val_features_list: list, val_labels_list: list,
                           model_type: str, n_trials: int = 100,
                           timeout: int = 3600,
                           random_state: int = 42,
                           n_validation_realizations: int = 1,
                           n_splits: int = 5,
                           feature_engineer_config = None,
                           study_name: str = "aquaculture_optimization",
                           storage: Optional[str] = None) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Perform hyperparameter optimization using Optuna with Stratified K-Fold CV.

    Parameters
    ----------
    X_train : np.ndarray
        Full training features
    y_train : np.ndarray
        Full training targets
    val_features_list : list
        List of precomputed validation features for each fold
    val_labels_list : list
        List of validation labels for each fold
    model_type : str
        Type of model ('lightgbm', 'catboost', or 'xgboost')
    n_trials : int, default=100
        Number of optimization trials
    timeout : int, default=3600
        Timeout in seconds
    random_state : int, default=42
        Random seed for reproducibility
    n_validation_realizations : int, default=1
        Number of validation realizations to average over
    n_splits : int, default=5
        Number of CV folds
    feature_engineer_config : aquaculture.config.AquacultureConfig, optional
        Configuration for the feature engineering pipeline
    study_name : str, default="aquaculture_optimization"
        Name for the Optuna study
    storage : str, optional
        Database URL for persistent storage

    Returns
    -------
    tuple
        (study, best_params) where study is the Optuna study object and
        best_params is a dictionary of the best hyperparameters
    """
    # Create study
    study = create_optuna_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )

    # Create objective function
    objective = create_objective_function(
        X_train=X_train,
        y_train=y_train,
        val_features_list=val_features_list,
        val_labels_list=val_labels_list,
        model_type=model_type,
        random_state=random_state,
        n_validation_realizations=n_validation_realizations,
        n_splits=n_splits,
        feature_engineer_config=feature_engineer_config
    )

    # Optimize
    logger.info(f"Starting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # Log results
    logger.info(f"Optimization completed!")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Analyze CV trends
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        cv_mean_scores = [t.user_attrs.get("cv_mean_score", 0) for t in completed_trials]
        cv_std_scores = [t.user_attrs.get("cv_std_score", 0) for t in completed_trials]

        if cv_mean_scores:
            avg_cv_mean_score = np.mean(cv_mean_scores)
            avg_cv_std_score = np.mean(cv_std_scores)
            logger.info(f"Average CV mean score: {avg_cv_mean_score:.4f}")
            logger.info(f"Average CV std score: {avg_cv_std_score:.4f}")

            # Collect fold scores for additional analysis
            fold_scores_dict = {}
            for i in range(n_splits):
                fold_key = f"fold_{i}_score"
                fold_scores = [t.user_attrs.get(fold_key, 0) for t in completed_trials]
                if fold_scores:
                    fold_scores_dict[f"fold_{i}"] = {
                        "mean": np.mean(fold_scores),
                        "std": np.std(fold_scores),
                        "min": np.min(fold_scores),
                        "max": np.max(fold_scores)
                    }

            # Log fold statistics
            for fold_id, stats in fold_scores_dict.items():
                logger.info(f"{fold_id} - Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, "
                          f"Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")

    return study, study.best_params


def save_study(study: optuna.Study, filepath: Union[str, Path]) -> None:
    """
    Save an Optuna study to disk.

    Parameters
    ----------
    study : optuna.Study
        The study to save
    filepath : str or Path
        Path where to save the study
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, filepath)
    logger.info(f"Study saved to {filepath}")


def load_study(filepath: Union[str, Path]) -> optuna.Study:
    """
    Load an Optuna study from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved study

    Returns
    -------
    optuna.Study
        Loaded study
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Study file not found: {filepath}")
    study = joblib.load(filepath)
    logger.info(f"Study loaded from {filepath}")
    return study


def get_parameter_importance(study: optuna.Study) -> Dict[str, float]:
    """
    Get parameter importance from an Optuna study.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study

    Returns
    -------
    dict
        Dictionary mapping parameter names to importance scores
    """
    try:
        importance = optuna.importance.get_param_importances(study)
        logger.info("Parameter importance calculated")
        return importance
    except Exception as e:
        logger.warning(f"Could not calculate parameter importance: {e}")
        return {}


def plot_optimization_history(study: optuna.Study, save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot optimization history.

    Parameters
    ----------
    study : optuna.Study
        Optuna study to plot
    save_path : str or Path, optional
        Path to save the plot. If None, plot is displayed
    """
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Optimization history plot saved to {save_path}")
        else:
            plt.show()
    except ImportError:
        logger.warning("Matplotlib not available for plotting")


def plot_param_importances(study: optuna.Study, save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot parameter importances.

    Parameters
    ----------
    study : optuna.Study
        Optuna study to plot
    save_path : str or Path, optional
        Path to save the plot. If None, plot is displayed
    """
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Parameter importance plot saved to {save_path}")
        else:
            plt.show()
    except ImportError:
        logger.warning("Matplotlib not available for plotting")


def plot_training_vs_validation_scores(study: optuna.Study, save_path: Optional[Union[str, Path]] = None,
                                      figsize: tuple = (10, 6)) -> None:
    """
    Plot training vs validation scores across optimization trials to detect overfitting.

    Parameters
    ----------
    study : optuna.Study
        Optuna study to plot
    save_path : str or Path, optional
        Path to save the plot. If None, plot is displayed
    figsize : tuple, default=(10, 6)
        Figure size
    """
    try:
        import matplotlib.pyplot as plt

        # Extract trial numbers and scores
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            logger.warning("No completed trials found for plotting")
            return

        trial_numbers = [t.number for t in trials]
        train_scores = [t.user_attrs.get("train_score", 0) for t in trials]
        val_scores = [t.user_attrs.get("val_score", 0) for t in trials]
        overfitting_gaps = [t.user_attrs.get("overfitting_gap", 0) for t in trials]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot training vs validation scores
        ax1.plot(trial_numbers, train_scores, 'b-o', label='Training Score', alpha=0.7, linewidth=1)
        ax1.plot(trial_numbers, val_scores, 'r-s', label='Validation Score', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Competition Score')
        ax1.set_title('Training vs Validation Scores Across Optimization Trials')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot overfitting gap
        ax2.bar(trial_numbers, overfitting_gaps, alpha=0.7, color='orange', label='Overfitting Gap (Train-Val)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Warning Threshold (0.05)')
        ax2.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Danger Threshold (0.1)')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Overfitting Gap')
        ax2.set_title('Overfitting Gap Across Optimization Trials')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Training vs validation scores plot saved to {save_path}")
        else:
            plt.show()

    except ImportError:
        logger.warning("Matplotlib not available for plotting")
    except Exception as e:
        logger.warning(f"Error creating training vs validation plot: {e}")


def plot_optimization_history_with_training(study: optuna.Study, save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot optimization history including both training and validation scores.

    Parameters
    ----------
    study : optuna.Study
        Optuna study to plot
    save_path : str or Path, optional
        Path to save the plot. If None, plot is displayed
    """
    try:
        import matplotlib.pyplot as plt

        # Get standard optimization history
        ax = optuna.visualization.matplotlib.plot_optimization_history(study)

        # Add training scores to the plot if available
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if trials:
            train_scores = [t.user_attrs.get("train_score", 0) for t in trials]
            trial_numbers = [t.number for t in trials]

            # Add training score line to the axes
            ax.plot(trial_numbers, train_scores, 'g--', alpha=0.7, label='Training Score', linewidth=1)
            ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Enhanced optimization history plot saved to {save_path}")
        else:
            plt.show()
    except ImportError:
        logger.warning("Matplotlib not available for plotting")
    except Exception as e:
        logger.warning(f"Error creating enhanced optimization history plot: {e}")


def plot_metrics_vs_trials(study: optuna.Study, metric_names: list, save_path: Optional[Union[str, Path]] = None,
                          figsize: tuple = (10, 6)) -> None:
    """
    Plot specified metrics across optimization trials.

    Parameters
    ----------
    study : optuna.Study
        Optuna study to plot
    metric_names : list of str
        List of metric names to extract from trial user_attrs (e.g., ['train_score', 'val_score', 'test_score'])
    save_path : str or Path, optional
        Path to save the plot. If None, plot is displayed
    figsize : tuple, default=(10, 6)
        Figure size
    """
    try:
        import matplotlib.pyplot as plt

        # Extract trial numbers and metrics
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            logger.warning("No completed trials found for plotting")
            return

        trial_numbers = [t.number for t in trials]

        fig, ax = plt.subplots(figsize=figsize)

        for metric in metric_names:
            metric_values = [t.user_attrs.get(metric, None) for t in trials]
            # Filter out None values (if metric missing in some trials)
            valid_indices = [i for i, v in enumerate(metric_values) if v is not None]
            if valid_indices:
                valid_trial_numbers = [trial_numbers[i] for i in valid_indices]
                valid_metric_values = [metric_values[i] for i in valid_indices]
                ax.plot(valid_trial_numbers, valid_metric_values, 'o-', label=metric, alpha=0.7, linewidth=1)

        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Metric Value')
        ax.set_title('Metrics Across Optimization Trials')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Metrics vs trials plot saved to {save_path}")
        else:
            plt.show()
    except ImportError:
        logger.warning("Matplotlib not available for plotting")
    except Exception as e:
        logger.warning(f"Error creating metrics vs trials plot: {e}")

def plot_all_metrics_vs_trials(
    study: optuna.Study,
    metric_prefixes: Sequence[str] = ("train_", "val_"),
    ignore: Sequence[str] = ("competition_score", "overfitting_gap"),
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot every metric stored in trial.user_attrs that matches the given prefixes.
    By default it plots all ``train_*`` and ``val_*`` metrics (except the ones
    listed in ``ignore``).

    Parameters
    ----------
    study : optuna.Study
        The Optuna study to visualise.
    metric_prefixes : tuple of str, optional
        Prefixes that identify which attributes to plot (default: training and validation).
    ignore : tuple of str, optional
        Attribute names to skip even if they match a prefix.
    figsize : tuple, optional
        Size of the matplotlib figure.
    save_path : str or Path, optional
        If supplied, the figure is saved to this path; otherwise it is shown interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available for plotting")
        return

    # Gather completed trials
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        logger.warning("No completed trials found for plotting")
        return

    trial_numbers = [t.number for t in trials]

    # Discover all attribute names that match the prefixes and are not ignored
    attr_names = set()
    for t in trials:
        for k in t.user_attrs.keys():
            if any(k.startswith(p) for p in metric_prefixes) and k not in ignore:
                attr_names.add(k)

    if not attr_names:
        logger.warning("No matching metrics found to plot")
        return

    attr_names = sorted(attr_names)  # deterministic order
    n_metrics = len(attr_names)

    # Create a grid of subplots – we try to keep it roughly square
    ncols = int(math.ceil(math.sqrt(n_metrics)))
    nrows = int(math.ceil(n_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, metric in enumerate(attr_names):
        ax = axes[idx]
        values = [t.user_attrs.get(metric, float("nan")) for t in trials]
        ax.plot(trial_numbers, values, "o-", alpha=0.7)
        ax.set_title(metric)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Metrics vs. Trial Number", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"All‑metrics plot saved to {save_path}")
    else:
        plt.show()