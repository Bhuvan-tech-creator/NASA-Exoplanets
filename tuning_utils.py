import numpy as np
from typing import Dict, Any, Tuple

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb


def _stratified_cv_indices(y: np.ndarray, n_splits: int, random_state: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros_like(y), y))


def tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Tune RandomForest hyperparameters with Optuna to maximize accuracy."""

    cv_indices = _stratified_cv_indices(y, n_splits, random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 8, 36),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            # Force bootstrap to enable max_samples subsampling for speed
            "bootstrap": True,
            "max_samples": trial.suggest_float("max_samples", 0.6, 1.0),
            "random_state": random_state,
            "n_jobs": -1,
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        }

        accs = []
        for train_idx, val_idx in cv_indices:
            clf = RandomForestClassifier(**params)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[val_idx])
            accs.append(accuracy_score(y[val_idx], pred))

        return float(np.mean(accs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _calc_scale_pos_weight(y: np.ndarray) -> float:
    # avoid div by zero
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0:
        return 1.0
    return float(neg / max(1, pos))


def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 30,
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters with Optuna using early stopping and stratified CV."""

    cv_indices = _stratified_cv_indices(y, n_splits, random_state)
    scale_pos_weight = _calc_scale_pos_weight(y)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": random_state,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
            "tree_method": "hist",
            "scale_pos_weight": scale_pos_weight,
        }

        accs = []
        for train_idx, val_idx in cv_indices:
            clf = xgb.XGBClassifier(**params)
            clf.fit(X[train_idx], y[train_idx])
            pred = (clf.predict_proba(X[val_idx])[:, 1] > 0.5).astype(int)
            accs.append(accuracy_score(y[val_idx], pred))

        return float(np.mean(accs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    # Ensure required fixed params preserved
    best_params.update({
        "random_state": random_state,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight,
    })
    return best_params


def tune_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 30,
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Tune LightGBM hyperparameters with Optuna using early stopping and stratified CV."""

    cv_indices = _stratified_cv_indices(y, n_splits, random_state)
    scale_pos_weight = _calc_scale_pos_weight(y)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "random_state": random_state,
            "objective": "binary",
            "class_weight": None,
            "n_jobs": -1,
            "verbose": -1,
            "scale_pos_weight": scale_pos_weight,
        }

        accs = []
        for train_idx, val_idx in cv_indices:
            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X[train_idx],
                y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            pred = (clf.predict_proba(X[val_idx])[:, 1] > 0.5).astype(int)
            accs.append(accuracy_score(y[val_idx], pred))

        return float(np.mean(accs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({
        "random_state": random_state,
        "objective": "binary",
        "n_jobs": -1,
        "verbose": -1,
        "scale_pos_weight": scale_pos_weight,
    })
    return best_params
