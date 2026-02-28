import pandas as pd
from src.utils.common import load_yaml, get_logger
logger = get_logger(__name__)
EXPECTED_COLUMNS = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate', 'Reporting_Airline', 'DOT_ID_Reporting_Airline', 'IATA_CODE_Reporting_Airline', 'Tail_Number', 'Flight_Number_Reporting_Airline', 'OriginAirportID', 'OriginAirportSeqID', 'OriginCityMarketID', 'Origin', 'OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName', 'OriginWac', 'DestAirportID', 'DestAirportSeqID', 'DestCityMarketID', 'Dest', 'DestCityName', 'DestState', 'DestStateFips', 'DestStateName', 'DestWac', 'CRSDepTime', 'DepTime', 'DepDelay', 'DepDelayMinutes', 'DepDel15', 'DepartureDelayGroups', 'DepTimeBlk', 'TaxiOut', 'WheelsOff', 'WheelsOn', 'TaxiIn', 'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDelayMinutes', 'ArrDel15', 'ArrivalDelayGroups', 'ArrTimeBlk', 'Cancelled', 'CancellationCode', 'Diverted', 'CRSElapsedTime', 'ActualElapsedTime', 'AirTime', 'Flights', 'Distance', 'DistanceGroup', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

def check_expected_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning('Missing %d expected column(s): %s', len(missing), missing)
    else:
        logger.info('All %d expected columns present.', len(EXPECTED_COLUMNS))
    return missing

def audit_nulls(df: pd.DataFrame) -> pd.Series:
    null_pct = df.isnull().mean().sort_values(ascending=False)
    high_nulls = null_pct[null_pct > 0.5]
    if not high_nulls.empty:
        logger.info('%d column(s) have >50%% nulls — candidates for removal.', len(high_nulls))
    return null_pct

def drop_high_null_columns(df: pd.DataFrame, threshold: float=0.6) -> pd.DataFrame:
    null_pct = df.isnull().mean()
    to_drop = null_pct[null_pct > threshold].index.tolist()
    if to_drop:
        logger.info('Dropping %d high-null columns: %s', len(to_drop), to_drop)
        df = df.drop(columns=to_drop)
    return df

def validate_data(df: pd.DataFrame, config: dict | None=None) -> pd.DataFrame:
    if config is None:
        config = load_yaml()
    threshold = config['features'].get('drop_threshold', 0.6)
    check_expected_columns(df)
    audit_nulls(df)
    df = drop_high_null_columns(df, threshold=threshold)
    logger.info('Validation complete — %d rows × %d columns remain.', *df.shape)
    return df
