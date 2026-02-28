"""
trainer.py
~~~~~~~~~~
Model training with hyperparameter search.

Trains a couple of classifiers (RandomForest and GradientBoosting),
picks the best one by F1-score, and serialises it to disk.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.utils.common import load_yaml, get_logger, ensure_dir, MODELS_DIR

logger = get_logger(__name__)


def _build_candidates(config: dict) -> list[tuple[str, object, dict]]:
    """Construct a list of (name, estimator, param_grid) tuples from config.

    We keep this as a plain function (not a class) so it's dead-simple to
    extend — just add another entry to config.yaml and mirror it here.
    """
    model_cfg = config["model"]
    candidates = []

    # ── Random Forest ─────────────────────────────────────
    rf_params = model_cfg.get("random_forest", {})
    candidates.append((
        "RandomForest",
        RandomForestClassifier(random_state=model_cfg["random_state"]),
        {
            "n_estimators":    rf_params.get("n_estimators", [100]),
            "max_depth":       [d if d is not None else None for d in rf_params.get("max_depth", [None])],
            "min_samples_split": rf_params.get("min_samples_split", [2]),
        },
    ))

    # ── Gradient Boosting ─────────────────────────────────
    gb_params = model_cfg.get("gradient_boosting", {})
    candidates.append((
        "GradientBoosting",
        GradientBoostingClassifier(random_state=model_cfg["random_state"]),
        {
            "n_estimators":  gb_params.get("n_estimators", [100]),
            "learning_rate": gb_params.get("learning_rate", [0.1]),
            "max_depth":     gb_params.get("max_depth", [3]),
        },
    ))

    return candidates


def train_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict | None = None,
) -> tuple[object, str, dict]:
    """Train multiple classifiers via randomised search and return the best.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data produced by ``transform_data``.
    config : dict, optional

    Returns
    -------
    tuple[estimator, model_name, best_params]
        The fitted estimator, its human-readable name, and the winning
        hyperparameters.
    """
    if config is None:
        config = load_yaml()

    random_state = config["model"]["random_state"]
    best_score   = -1.0
    best_model   = None
    best_name    = ""
    best_params  = {}

    for name, estimator, param_grid in _build_candidates(config):
        logger.info("Training %s …", name)

        search = RandomizedSearchCV(
            estimator,
            param_distributions=param_grid,
            n_iter=min(6, _grid_size(param_grid)),   # keep it fast
            scoring="f1",
            cv=3,
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
        )
        search.fit(X_train, y_train)

        logger.info(
            "%s — best CV F1: %.4f  |  params: %s",
            name,
            search.best_score_,
            search.best_params_,
        )

        if search.best_score_ > best_score:
            best_score  = search.best_score_
            best_model  = search.best_estimator_
            best_name   = name
            best_params = search.best_params_

    # Persist the winner
    ensure_dir(MODELS_DIR)
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info("Saved best model (%s) → %s", best_name, model_path)

    return best_model, best_name, best_params


def _grid_size(param_grid: dict) -> int:
    """Rough count of combinations in the grid."""
    size = 1
    for values in param_grid.values():
        size *= len(values) if isinstance(values, list) else 1
    return size
