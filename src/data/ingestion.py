import os
import zipfile
from pathlib import Path
import pandas as pd
from src.utils.common import load_yaml, get_logger, ensure_dir
logger = get_logger(__name__)

def unzip_archive(archive_path: Path, extract_to: Path) -> list[Path]:
    ensure_dir(extract_to)
    logger.info('Extracting %s → %s', archive_path, extract_to)
    with zipfile.ZipFile(archive_path, 'r') as zf:
        zf.extractall(extract_to)
        extracted = [extract_to / name for name in zf.namelist()]
    logger.info('Extracted %d file(s).', len(extracted))
    return extracted

def find_csv(directory: Path) -> Path:
    csv_files = sorted(directory.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {directory}')
    chosen = max(csv_files, key=lambda p: p.stat().st_size)
    logger.info('Selected CSV: %s (%.1f MB)', chosen.name, chosen.stat().st_size / 1000000.0)
    return chosen

def ingest_data(config: dict | None=None) -> pd.DataFrame:
    if config is None:
        config = load_yaml()
    data_cfg = config['data']
    archive_path = Path(data_cfg['raw_data_dir']) / data_cfg['archive_name']
    extract_dir = Path(data_cfg['extract_dir'])
    if not extract_dir.exists() or not any(extract_dir.glob('*.csv')):
        if not archive_path.exists():
            raise FileNotFoundError(f'Archive not found at {archive_path}. Download it from Kaggle and place it in the data directory.')
        unzip_archive(archive_path, extract_dir)
    csv_path = find_csv(extract_dir)
    logger.info('Loading CSV into DataFrame …')
    try:
        df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed, falling back to 'latin1' encoding...")
        df = pd.read_csv(csv_path, low_memory=False, encoding='latin1', on_bad_lines='skip')
    sample_size = data_cfg.get('sample_size')
    if sample_size and sample_size < len(df):
        logger.info('Sampling %d rows from %d based on config.', sample_size, len(df))
        df = df.sample(n=sample_size, random_state=42)
    logger.info('Loaded %d rows × %d columns.', *df.shape)
    return df
