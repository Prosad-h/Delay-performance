import os
import logging
import yaml
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / 'config' / 'config.yaml'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'
DATA_DIR = Path('D:/Project/data')

def load_yaml(path: Path=CONFIG_PATH) -> dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)

def get_logger(name: str, log_file: str='service.log') -> logging.Logger:
    ensure_dir(LOGS_DIR)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)-8s %(name)s — %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(LOGS_DIR / log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
