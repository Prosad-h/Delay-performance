from src.utils.common import load_yaml, get_logger
from src.data.ingestion import ingest_data
from src.data.validation import validate_data
from src.data.transformation import transform_data
from src.model.trainer import train_best_model
from src.model.evaluator import evaluate_model
from src.experiment.tracking import init_tracking, log_experiment
logger = get_logger(__name__)

def run_training_pipeline() -> None:
    logger.info('=' * 60)
    logger.info('STARTING TRAINING PIPELINE')
    logger.info('=' * 60)
    config = load_yaml()
    logger.info('Step 1/6 — Ingestion')
    raw_df = ingest_data(config)
    logger.info('Step 2/6 — Validation')
    clean_df = validate_data(raw_df, config)
    logger.info('Step 3/6 — Transformation')
    data = transform_data(clean_df, config)
    logger.info('Step 4/6 — Training')
    best_model, model_name, best_params = train_best_model(data['X_train'], data['y_train'], config)
    logger.info('Step 5/6 — Evaluation')
    metrics = evaluate_model(best_model, data['X_test'], data['y_test'])
    logger.info('Step 6/6 — Logging to DagsHub / MLflow')
    init_tracking(config)
    run_id = log_experiment(model_name=model_name, params=best_params, metrics=metrics, model=best_model)
    logger.info('=' * 60)
    logger.info('PIPELINE COMPLETE  —  MLflow run ID: %s', run_id)
    logger.info('=' * 60)
if __name__ == '__main__':
    run_training_pipeline()
