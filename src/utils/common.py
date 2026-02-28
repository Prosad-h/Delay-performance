"""
common.py
~~~~~~~~~
Shared helper functions used across the entire project.

Provides:
    - YAML config loader
    - Standardised logger factory
    - Directory-creation helper
    - Path constants so every module resolves paths consistently
"""

import os
import logging
import yaml
from pathlib import Path


# ── Path constants ────────────────────────────────────────
# PROJECT_ROOT points to the top-level "Python" directory so that every
# module can resolve relative paths without fragile hard-coding.
PROJECT_ROOT = Path(__file__).resolve().parents[2]    # …/Python/
CONFIG_PATH  = PROJECT_ROOT / "config" / "config.yaml"
MODELS_DIR   = PROJECT_ROOT / "models"
LOGS_DIR     = PROJECT_ROOT / "logs"
DATA_DIR     = Path("D:/Project/data")


def load_yaml(path: Path = CONFIG_PATH) -> dict:
    """Read a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the YAML file.
        Defaults to ``config/config.yaml`` at the project root.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_logger(name: str, log_file: str = "service.log") -> logging.Logger:
    """Create (or retrieve) a logger with both console and file handlers.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__`` from the calling module.
    log_file : str
        Filename inside the ``logs/`` directory.

    Returns
    -------
    logging.Logger
    """
    ensure_dir(LOGS_DIR)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the function is called more
    # than once for the same logger name.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO and above)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (DEBUG and above — captures everything)
    file_handler = logging.FileHandler(LOGS_DIR / log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def ensure_dir(path: Path) -> Path:
    """Create a directory (and any parents) if it doesn't already exist.

    Parameters
    ----------
    path : Path
        Directory path to create.

    Returns
    -------
    Path
        The same path, for convenient chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
