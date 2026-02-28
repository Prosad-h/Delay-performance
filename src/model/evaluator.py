"""
evaluator.py
~~~~~~~~~~~~
Computes classification metrics on the held-out test set.

Returns everything as a plain dict so callers (pipeline, MLflow logger)
can consume the metrics without coupling to sklearn's report format.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from src.utils.common import get_logger

logger = get_logger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Run predictions on *X_test* and compute standard metrics.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test, y_test : np.ndarray

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, roc_auc,
              classification_report (as a formatted string).
    """
    y_pred = model.predict(X_test)

    # ROC-AUC needs probability estimates.  Fall back gracefully if the
    # estimator doesn't support predict_proba (unlikely with our models,
    # but just being safe).
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except (AttributeError, IndexError):
        roc_auc = None
        logger.warning("predict_proba unavailable — skipping ROC-AUC.")

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc,
    }

    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info("Classification report:\n%s", report)

    for key, value in metrics.items():
        if value is not None:
            logger.info("  %-12s %.4f", key, value)

    metrics["classification_report"] = report
    return metrics
