"""
schemas.py
~~~~~~~~~~
Lightweight request / response validation for the prediction endpoint.

We keep this dependency-free (no Pydantic or Marshmallow) to stay light
for Railway deployment.  The ``validate_prediction_request`` function
returns a clean dict or raises ``ValueError`` with a human-readable
message.
"""

# The fields the /predict endpoint expects.  This mirrors the feature
# list in config.yaml — if you add features there, add them here too.
REQUIRED_FIELDS = {
    "Month":              int,
    "DayOfWeek":          int,
    "CRSDepTime":         (int, float),
    "CRSArrTime":         (int, float),
    "CRSElapsedTime":     (int, float),
    "Distance":           (int, float),
    "DepDelay":           (int, float),
    "Reporting_Airline":  str,
    "Origin":             str,
    "Dest":               str,
    "DepTimeBlk":         str,
    "ArrTimeBlk":         str,
}


def validate_prediction_request(payload: dict | None) -> dict:
    """Validate and coerce an incoming JSON payload for /predict.

    Parameters
    ----------
    payload : dict
        Raw JSON body from the Flask request.

    Returns
    -------
    dict
        Cleaned payload with correct types.

    Raises
    ------
    ValueError
        If required fields are missing or have the wrong type.
    """
    if not payload or not isinstance(payload, dict):
        raise ValueError("Request body must be a non-empty JSON object.")

    cleaned = {}
    errors  = []

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in payload:
            errors.append(f"Missing required field: '{field}'.")
            continue

        value = payload[field]

        # Allow loose numeric coercion (e.g. "100" → 100)
        if expected_type in (int, float, (int, float)):
            try:
                value = float(value) if isinstance(expected_type, tuple) else expected_type(value)
            except (TypeError, ValueError):
                errors.append(
                    f"Field '{field}' must be numeric, got {type(value).__name__}."
                )
                continue
        elif expected_type is str:
            value = str(value)

        cleaned[field] = value

    if errors:
        raise ValueError(" | ".join(errors))

    return cleaned
