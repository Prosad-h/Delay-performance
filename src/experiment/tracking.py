"""
tracking.py
~~~~~~~~~~~
DagsHub + MLflow integration.

All experiment-tracking logic lives here so the rest of the codebase
never imports ``mlflow`` or ``dagshub`` directly.  This keeps coupling
low and makes it trivial to swap providers later.

Environment variables required (set in ``.env`` or Railway dashboard):
    - DAGSHUB_USERNAME
    - DAGSHUB_TOKEN
"""

import os
from dotenv import load_dotenv

import dagshub
import mlflow

from src.utils.common import load_yaml, get_logger

logger = get_logger(__name__)

# Load .env once at import time (no-op if the file doesn't exist).
load_dotenv()


def init_tracking(config: dict | None = None) -> None:
    """Initialise DagsHub + MLflow for the current process.

    Must be called **once** before any ``mlflow.start_run`` calls.
    """
    if config is None:
        config = load_yaml()

    mlflow_cfg = config["mlflow"]

    username = os.getenv("DAGSHUB_USERNAME", mlflow_cfg.get("dagshub_repo_owner", ""))
    token    = os.getenv("DAGSHUB_TOKEN", "")

    if token:
        # Set credentials so the MLflow client can authenticate.
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # dagshub.init sets MLFLOW_TRACKING_URI behind the scenes, but we
    # set it explicitly as well for clarity and to guarantee consistency.
    dagshub.init(
        repo_owner=mlflow_cfg["dagshub_repo_owner"],
        repo_name=mlflow_cfg["dagshub_repo_name"],
        mlflow=True,
    )

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    logger.info(
        "MLflow tracking initialised — URI: %s  |  experiment: %s",
        mlflow_cfg["tracking_uri"],
        mlflow_cfg["experiment_name"],
    )


def log_experiment(
    model_name: str,
    params: dict,
    metrics: dict,
    model=None,
    artifacts: dict | None = None,
) -> str:
    """Log a single training run to MLflow.

    Parameters
    ----------
    model_name : str
        Human-readable model label (e.g. "RandomForest").
    params : dict
        Hyperparameters to record.
    metrics : dict
        Evaluation metrics (accuracy, f1, etc.).
    model : sklearn estimator, optional
        If provided, the model is logged as an MLflow artefact via
        ``mlflow.sklearn.log_model``.
    artifacts : dict, optional
        Extra files to log, keyed by artefact name → local path.

    Returns
    -------
    str
        The MLflow run ID.
    """
    with mlflow.start_run() as run:
        mlflow.set_tag("model_type", model_name)

        # Log hyperparameters
        mlflow.log_params(params)

        # Log metrics (skip non-numeric values like the text report)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(key, value)

        # Log the model artefact
        if model is not None:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # Log any extra files (e.g. classification report as a text file)
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=name)

        logger.info(
            "Logged run %s for model '%s'  [F1=%.4f]",
            run.info.run_id,
            model_name,
            metrics.get("f1", 0),
        )
        return run.info.run_id
