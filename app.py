import os
from pathlib import Path
import joblib
from flask import Flask
from flask_cors import CORS
from src.api.routes import api_bp
from src.utils.common import load_yaml, get_logger, MODELS_DIR
logger = get_logger(__name__)

def create_app() -> Flask:
    application = Flask(__name__)
    CORS(application)
    config = load_yaml()
    model_path = MODELS_DIR / 'best_model.joblib'
    encoders_path = MODELS_DIR / 'encoders.joblib'
    scaler_path = MODELS_DIR / 'scaler.joblib'
    features_path = MODELS_DIR / 'feature_names.joblib'
    if model_path.exists():
        application.config['MODEL'] = joblib.load(model_path)
        application.config['ENCODERS'] = joblib.load(encoders_path)
        application.config['SCALER'] = joblib.load(scaler_path)
        application.config['FEATURE_NAMES'] = joblib.load(features_path)
        logger.info('Loaded model and preprocessors from %s.', MODELS_DIR)
    else:
        logger.warning('No trained model found at %s. Run the training pipeline first: python -m src.pipeline.run_pipeline', model_path)
        application.config['MODEL'] = None
        application.config['ENCODERS'] = None
        application.config['SCALER'] = None
        application.config['FEATURE_NAMES'] = None
    application.register_blueprint(api_bp)
    return application
app = create_app()
if __name__ == '__main__':
    config = load_yaml()
    server = config.get('server', {})
    app.run(host=server.get('host', '0.0.0.0'), port=int(os.environ.get('PORT', server.get('port', 5000))), debug=server.get('debug', False))
