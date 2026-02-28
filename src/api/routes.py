"""
routes.py
~~~~~~~~~
Flask Blueprint containing the API endpoints.

Endpoints
---------
GET  /health   — simple liveness / readiness probe.
POST /predict  — accepts flight features as JSON, returns delay prediction.
"""

import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify

from src.api.schemas import validate_prediction_request
from src.utils.common import get_logger

logger = get_logger(__name__)

# The Blueprint is registered in app.py — we don't import the Flask
# app object here, keeping things nicely decoupled.
api_bp = Blueprint("api", __name__)


# ── Health check ──────────────────────────────────────────
@api_bp.route("/health", methods=["GET"])
def health():
    """Return a simple JSON heartbeat.

    Used by Railway (and any upstream load balancer) to confirm the
    service is up.
    """
    return jsonify({"status": "healthy"}), 200


# ── Prediction ────────────────────────────────────────────
@api_bp.route("/predict", methods=["POST"])
def predict():
    """Accept flight features and return a delay prediction.

    Expected JSON body (example)::

        {
            "Month": 6,
            "DayOfWeek": 3,
            "CRSDepTime": 1430,
            "CRSArrTime": 1715,
            "CRSElapsedTime": 165,
            "Distance": 967,
            "DepDelay": 12.0,
            "Reporting_Airline": "AA",
            "Origin": "DFW",
            "Dest": "LAX",
            "DepTimeBlk": "1400-1459",
            "ArrTimeBlk": "1700-1759"
        }

    Response::

        {
            "delay_prediction": 0,
            "probability": 0.23
        }
    """
    from flask import current_app

    try:
        payload = validate_prediction_request(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Retrieve the model and preprocessors that were loaded at app startup.
    model         = current_app.config["MODEL"]
    encoders      = current_app.config["ENCODERS"]
    scaler        = current_app.config["SCALER"]
    feature_names = current_app.config["FEATURE_NAMES"]

    # Build a single-row DataFrame matching the training feature order.
    row = pd.DataFrame([payload])

    # -- Encode categoricals (same logic as transformation.py) ----------
    from src.data.transformation import encode_categoricals, scale_numerics
    from src.utils.common import load_yaml

    config    = load_yaml()
    feat_cfg  = config["features"]

    row, _ = encode_categoricals(
        row,
        feat_cfg["categorical"],
        encoders=encoders,
        fit=False,
    )
    row, _ = scale_numerics(
        row,
        feat_cfg["numeric"],
        scaler=scaler,
        fit=False,
    )

    # Ensure column order matches what the model was trained on.
    row = row.reindex(columns=feature_names, fill_value=0)

    # -- Predict --------------------------------------------------------
    prediction = int(model.predict(row.values)[0])

    proba = None
    try:
        proba = float(model.predict_proba(row.values)[0][1])
    except AttributeError:
        pass  # estimator doesn't support predict_proba

    result = {"delay_prediction": prediction}
    if proba is not None:
        result["probability"] = round(proba, 4)

    logger.info("Prediction: %s  |  input: %s", result, payload)
    return jsonify(result), 200
