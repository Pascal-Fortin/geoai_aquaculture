"""
Optuna utilities for hyperparameter optimization in the aquaculture machine learning framework.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Callable, Tuple
import optuna
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import joblib

from .model_factory import ModelFactory
from .metrics import competition_score
from .evaluate import cross_validate_model

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
                            X_val: np.ndarray, y_val: np.ndarray,
                            model_type: str, random_state: int = 42,
                            n_validation_realizations: int = 1) -> Callable[[optuna.Trial], float]:
    """
    Create an objective function for Optuna optimization.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation targets
    model_type : str
        Type of model ('lightgbm', 'catboost', or 'xgboost')
    random_state : int, default=42
        Random seed for reproducibility
    n_validation_realizations : int, default=1
        Number of validation realizations to average over

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

            # Handle multiple validation realizations
            if n_validation_realizations > 1:
                scores = []
                for i in range(n_validation_realizations):
                    # In a real implementation, we would apply different observation process realizations
                    # For now, we'll use the same validation set but this could be extended
                    model.fit(X_train, y_train)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    score = competition_score(y_val, y_pred_proba)
                    scores.append(score)
                final_score = np.mean(scores)
            else:
                # Standard single validation realization
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                final_score = competition_score(y_val, y_pred_proba)

            # Report intermediate value for pruning
            trial.report(final_score, step=1)

            # Handle pruning based on intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()

            return final_score

        except Exception as e:
            logger.warning(f"Trial failed with error: {e}")
            # Return a very poor score to discourage this parameter setting
            return 0.0

    return objective


def optimize_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           model_type: str, n_trials: int = 100,
                           timeout: int = 3600,
                           random_state: int = 42,
                           n_validation_realizations: int = 1,
                           study_name: str = "aquaculture_optimization",
                           storage: Optional[str] = None) -> Tuple[optuna.Study, Dict[str, Any]]:
    """
    Perform hyperparameter optimization using Optuna.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation targets
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
        X_val=X_val,
        y_val=y_val,
        model_type=model_type,
        random_state=random_state,
        n_validation_realizations=n_validation_realizations
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
        fig = optuna.visualization.matplotlib.plot_optimization_history(step)
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