"""
transformation.py
~~~~~~~~~~~~~~~~~
Feature engineering, encoding, scaling, and train/test splitting.

Takes the validated DataFrame and produces arrays (or DataFrames) that
are ready to be consumed by scikit-learn estimators.

Key decisions:
    - Target variable: ``ArrDel15`` (1 = arrival delay ≥ 15 min).
    - Rows where the target is NaN (cancelled / diverted flights) are
      dropped — we can't learn from flights that never arrived.
    - Categorical columns with a manageable cardinality are label-encoded.
    - Numeric features are standard-scaled.
    - Fitted encoders and the scaler are persisted to ``models/`` so the
      Flask API can re-use them at inference time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.common import load_yaml, get_logger, ensure_dir, MODELS_DIR

logger = get_logger(__name__)


def select_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Keep only the columns we actually need for modelling.

    Any column listed under ``features.numeric`` or ``features.categorical``
    in the config is retained, along with the target.
    """
    feat_cfg    = config["features"]
    target      = feat_cfg["target"]
    keep_cols   = feat_cfg["numeric"] + feat_cfg["categorical"] + [target]

    # Only keep columns that actually exist in the DataFrame — some may
    # have been dropped during validation.
    available   = [c for c in keep_cols if c in df.columns]
    missing     = set(keep_cols) - set(available)
    if missing:
        logger.warning("Requested features not available: %s", missing)

    return df[available].copy()


def drop_target_nulls(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Remove rows where the target is NaN (cancelled / diverted flights)."""
    before = len(df)
    df = df.dropna(subset=[target])
    after = len(df)
    logger.info(
        "Dropped %d rows with NaN target (%s). %d rows remain.",
        before - after,
        target,
        after,
    )
    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    categorical_cols : list[str]
        Column names to encode.
    encoders : dict, optional
        Pre-fitted ``LabelEncoder`` objects keyed by column name.
        Required when ``fit=False`` (inference mode).
    fit : bool
        If ``True``, fit new encoders and transform.  If ``False``,
        only transform using *encoders*.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, LabelEncoder]]
        The transformed DataFrame and the encoder dict.
    """
    if encoders is None:
        encoders = {}

    df = df.copy()

    for col in categorical_cols:
        if col not in df.columns:
            continue

        # Fill missing categoricals with a placeholder so the encoder
        # doesn't choke on NaNs.
        df[col] = df[col].fillna("__MISSING__").astype(str)

        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully at inference time: map
            # them to -1 rather than crashing.
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x, _k=known, _le=le: (
                _le.transform([x])[0] if x in _k else -1
            ))

    return df, encoders


def scale_numerics(
    df: pd.DataFrame,
    numeric_cols: list[str],
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standard-scale numeric features.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list[str]
    scaler : StandardScaler, optional
    fit : bool

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
    """
    if scaler is None:
        scaler = StandardScaler()

    df = df.copy()
    cols_present = [c for c in numeric_cols if c in df.columns]

    # Fill remaining numeric NaNs with column medians (safe default).
    for col in cols_present:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    if fit:
        df[cols_present] = scaler.fit_transform(df[cols_present])
    else:
        df[cols_present] = scaler.transform(df[cols_present])

    return df, scaler


def transform_data(
    df: pd.DataFrame,
    config: dict | None = None,
) -> dict:
    """Full transformation pipeline: select → clean → encode → scale → split.

    Returns a dict containing:
        - ``X_train``, ``X_test``, ``y_train``, ``y_test``  (numpy arrays)
        - ``feature_names``  (list[str])
        - ``encoders``       (dict of fitted LabelEncoders)
        - ``scaler``         (fitted StandardScaler)

    Fitted artefacts are also saved to ``models/`` as ``.joblib`` files.
    """
    if config is None:
        config = load_yaml()

    feat_cfg       = config["features"]
    target_col     = feat_cfg["target"]
    numeric_cols   = feat_cfg["numeric"]
    cat_cols       = feat_cfg["categorical"]
    test_size      = config["model"]["test_size"]
    random_state   = config["model"]["random_state"]

    # 1. Feature selection
    df = select_features(df, config)

    # 2. Drop rows with no target value
    df = drop_target_nulls(df, target_col)

    # 3. Separate target
    y = df[target_col].astype(int).values
    df = df.drop(columns=[target_col])

    # 4. Encode categoricals
    df, encoders = encode_categoricals(df, cat_cols)

    # 5. Scale numerics
    df, scaler = scale_numerics(df, numeric_cols)

    # 6. Train / test split (stratified to keep class balance)
    feature_names = df.columns.tolist()
    X = df.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split: train=%d, test=%d (%.0f%% test).",
        len(X_train),
        len(X_test),
        test_size * 100,
    )

    # 7. Persist fitted preprocessors for inference reuse
    ensure_dir(MODELS_DIR)
    joblib.dump(encoders, MODELS_DIR / "encoders.joblib")
    joblib.dump(scaler,   MODELS_DIR / "scaler.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")
    logger.info("Saved encoders, scaler, and feature names to %s.", MODELS_DIR)

    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_names": feature_names,
        "encoders": encoders,
        "scaler":   scaler,
    }
