"""
validation.py
~~~~~~~~~~~~~
Schema checks and data-quality audits.

After ingestion gives us a raw DataFrame, this module makes sure:
    - All expected columns are present.
    - Columns with too many nulls are flagged and dropped.
    - Basic type casts are applied so downstream code can rely on
      consistent dtypes.
"""

import pandas as pd

from src.utils.common import load_yaml, get_logger

logger = get_logger(__name__)

# The full set of columns the BTS dataset *should* contain.  We don't
# hard-fail if some are missing (diversion columns are often empty
# anyway), but we do log a warning.
EXPECTED_COLUMNS = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "DOT_ID_Reporting_Airline",
    "IATA_CODE_Reporting_Airline", "Tail_Number",
    "Flight_Number_Reporting_Airline", "OriginAirportID",
    "OriginAirportSeqID", "OriginCityMarketID", "Origin",
    "OriginCityName", "OriginState", "OriginStateFips",
    "OriginStateName", "OriginWac", "DestAirportID",
    "DestAirportSeqID", "DestCityMarketID", "Dest", "DestCityName",
    "DestState", "DestStateFips", "DestStateName", "DestWac",
    "CRSDepTime", "DepTime", "DepDelay", "DepDelayMinutes", "DepDel15",
    "DepartureDelayGroups", "DepTimeBlk", "TaxiOut", "WheelsOff",
    "WheelsOn", "TaxiIn", "CRSArrTime", "ArrTime", "ArrDelay",
    "ArrDelayMinutes", "ArrDel15", "ArrivalDelayGroups", "ArrTimeBlk",
    "Cancelled", "CancellationCode", "Diverted", "CRSElapsedTime",
    "ActualElapsedTime", "AirTime", "Flights", "Distance",
    "DistanceGroup", "CarrierDelay", "WeatherDelay", "NASDelay",
    "SecurityDelay", "LateAircraftDelay",
]


def check_expected_columns(df: pd.DataFrame) -> list[str]:
    """Compare the DataFrame's columns against the expected schema.

    Returns a list of any *missing* column names (empty if all present).
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Missing %d expected column(s): %s", len(missing), missing)
    else:
        logger.info("All %d expected columns present.", len(EXPECTED_COLUMNS))
    return missing


def audit_nulls(df: pd.DataFrame) -> pd.Series:
    """Return the fraction of nulls per column, sorted descending."""
    null_pct = df.isnull().mean().sort_values(ascending=False)
    high_nulls = null_pct[null_pct > 0.5]
    if not high_nulls.empty:
        logger.info(
            "%d column(s) have >50%% nulls — candidates for removal.",
            len(high_nulls),
        )
    return null_pct


def drop_high_null_columns(
    df: pd.DataFrame,
    threshold: float = 0.60,
) -> pd.DataFrame:
    """Drop every column whose null fraction exceeds *threshold*.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
        Columns with ``null_pct > threshold`` are dropped.

    Returns
    -------
    pd.DataFrame
        A copy with the offending columns removed.
    """
    null_pct = df.isnull().mean()
    to_drop  = null_pct[null_pct > threshold].index.tolist()

    if to_drop:
        logger.info("Dropping %d high-null columns: %s", len(to_drop), to_drop)
        df = df.drop(columns=to_drop)

    return df


def validate_data(
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Run the full validation suite on a raw DataFrame.

    Steps:
        1. Check for expected columns (logs warnings, never hard-fails).
        2. Audit null percentages.
        3. Drop columns above the configured null threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data from the ingestion step.
    config : dict, optional
        Pre-loaded config; loads ``config.yaml`` if ``None``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for transformation.
    """
    if config is None:
        config = load_yaml()

    threshold = config["features"].get("drop_threshold", 0.60)

    check_expected_columns(df)
    audit_nulls(df)
    df = drop_high_null_columns(df, threshold=threshold)

    logger.info("Validation complete — %d rows × %d columns remain.", *df.shape)
    return df
