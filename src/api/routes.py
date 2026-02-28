import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
from src.api.schemas import validate_prediction_request
from src.utils.common import get_logger
logger = get_logger(__name__)
api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health():
    return (jsonify({'status': 'healthy'}), 200)

@api_bp.route('/predict', methods=['POST'])
def predict():
    from flask import current_app
    try:
        payload = validate_prediction_request(request.get_json(silent=True))
    except ValueError as exc:
        return (jsonify({'error': str(exc)}), 400)
    model = current_app.config['MODEL']
    encoders = current_app.config['ENCODERS']
    scaler = current_app.config['SCALER']
    feature_names = current_app.config['FEATURE_NAMES']
    row = pd.DataFrame([payload])
    from src.data.transformation import encode_categoricals, scale_numerics
    from src.utils.common import load_yaml
    config = load_yaml()
    feat_cfg = config['features']
    row, _ = encode_categoricals(row, feat_cfg['categorical'], encoders=encoders, fit=False)
    row, _ = scale_numerics(row, feat_cfg['numeric'], scaler=scaler, fit=False)
    row = row.reindex(columns=feature_names, fill_value=0)
    prediction = int(model.predict(row.values)[0])
    proba = None
    try:
        proba = float(model.predict_proba(row.values)[0][1])
    except AttributeError:
        pass
    result = {'delay_prediction': prediction}
    if proba is not None:
        result['probability'] = round(proba, 4)
    logger.info('Prediction: %s  |  input: %s', result, payload)
    return (jsonify(result), 200)
