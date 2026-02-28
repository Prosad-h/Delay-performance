import os
from dotenv import load_dotenv
import dagshub
import mlflow
from src.utils.common import load_yaml, get_logger
logger = get_logger(__name__)
load_dotenv()

def init_tracking(config: dict | None=None) -> None:
    if config is None:
        config = load_yaml()
    mlflow_cfg = config['mlflow']
    username = os.getenv('DAGSHUB_USERNAME', mlflow_cfg.get('dagshub_repo_owner', ''))
    token = os.getenv('DAGSHUB_TOKEN', '')
    if token:
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    dagshub.init(repo_owner=mlflow_cfg['dagshub_repo_owner'], repo_name=mlflow_cfg['dagshub_repo_name'], mlflow=True)
    mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
    mlflow.set_experiment(mlflow_cfg['experiment_name'])
    logger.info('MLflow tracking initialised — URI: %s  |  experiment: %s', mlflow_cfg['tracking_uri'], mlflow_cfg['experiment_name'])

def log_experiment(model_name: str, params: dict, metrics: dict, model=None, artifacts: dict | None=None) -> str:
    with mlflow.start_run() as run:
        mlflow.set_tag('model_type', model_name)
        mlflow.log_params(params)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(key, value)
        if model is not None:
            mlflow.sklearn.log_model(model, artifact_path='model')
        if artifacts:
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=name)
        logger.info("Logged run %s for model '%s'  [F1=%.4f]", run.info.run_id, model_name, metrics.get('f1', 0))
        return run.info.run_id
