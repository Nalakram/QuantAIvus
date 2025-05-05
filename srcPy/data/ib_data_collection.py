import asyncio
from asyncio import Semaphore
from typing import Dict, List, Optional
import pandas as pd
from ib_insync import Stock, util, IB, BarData
from srcPy.utils.logger import logger
from srcPy.utils.validators import validate_symbol, validate_date
from srcPy.utils.exceptions import IBConnectionError, DataFetchError
from srcPy.utils.config import config
from pathlib import Path
from datetime import datetime

class NoDataError(DataFetchError):
    """Raised when no data is returned for a valid symbol."""
    def __init__(self, symbol: str):
        super().__init__(f"No historical data returned for {symbol}")

def _get_cache_path(symbol: str) -> Path:
    """Return cache file path for a symbol."""
    cache_dir = Path("data/raw/historical_prices_ib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}.parquet"

def _bars_to_df(bars: List[BarData]) -> pd.DataFrame:
    """
    Convert IB BarData list to a pandas DataFrame with a datetime index.

    Parameters:
        bars: List of BarData returned by IB.reqHistoricalData.

    Returns:
        pd.DataFrame indexed by 'date' with OHLCV columns.

    Raises:
        DataFetchError: If DataFrame conversion fails or data is invalid.
    """
    try:
        df = util.df(bars)
        if df.empty:
            raise DataFetchError("Empty DataFrame after conversion")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Validate expected columns
        expected_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not expected_columns.issubset(df.columns):
            raise DataFetchError(f"Missing expected columns: {expected_columns - set(df.columns)}")
        
        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='last')].sort_index()
        
        # Check for missing values
        if df[['open', 'high', 'low', 'close', 'volume']].isna().any().any():
            logger.warning("Missing values detected in DataFrame; filling with forward fill")
            df = df.fillna(method='ffill')
        
        return df
    except Exception as e:
        raise DataFetchError(f"Failed to convert bars to DataFrame: {str(e)}") from e

def create_mock_bars(n: int, start_date: str = "2025-01-01") -> List[BarData]:
    """Generate mock BarData for testing."""
    bars = []
    base_date = pd.to_datetime(start_date)
    for i in range(n):
        date = (base_date + pd.Timedelta(days=i)).strftime("%Y%m%d %H:%M:%S")
        bars.append(BarData(
            date=date,
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.5 + i,
            volume=1000 + i,
            barCount=1,
            average=100.25 + i
        ))
    return bars

async def _fetch_historical_async(
    symbol: str,
    end_date: str,
    duration: str,
    bar_size: str,
    ib: IB,
    sem: Semaphore,
    what_to_show: str,
    use_rth: bool,
    format_date: int,
    use_cache: bool
) -> pd.DataFrame:
    """
    Async helper to fetch historical data for a single symbol.

    Parameters:
        symbol: Stock ticker (e.g., 'AAPL').
        end_date: End date for data.
        duration: Duration string (e.g., '1 Y').
        bar_size: Granularity (e.g., '1 day').
        ib: IB instance.
        sem: Semaphore to limit concurrent requests.
        what_to_show: Data type (e.g., 'TRADES').
        use_rth: Restrict to regular trading hours.
        format_date: Date format for IB API.
        use_cache: If True, use and update cache.

    Returns:
        pd.DataFrame with historical data.

    Raises:
        ValueError: If symbol or date validation fails.
        IBConnectionError: For connection issues.
        NoDataError: If no data is returned.
        DataFetchError: For other errors.
    """
    validate_symbol(symbol)
    validate_date(end_date)
    log = logger.bind(symbol=symbol, duration=duration, end_date=end_date)
    log.info("Starting async historical data fetch")

    cache_path = _get_cache_path(symbol) if use_cache else None
    cached_df = None

    if use_cache and cache_path.exists():
        cached_df = pd.read_parquet(cache_path)
        if not cached_df.empty:
            last_date = cached_df.index.max()
            log.info(f"Found cached data up to {last_date}")
            duration = "1 D"  # Fetch only new data
            end_date = last_date.strftime("%Y%m%d %H:%M:%S")

    try:
        async with sem:
            bars = await ib.reqHistoricalDataAsync(
                Stock(symbol, 'SMART', 'USD'),
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=format_date
            )

        if not bars:
            if cached_df is not None and not cached_df.empty:
                log.info("No new data; returning cached data")
                return cached_df
            raise NoDataError(symbol)

        df = _bars_to_df(bars)
        if cached_df is not None and not cached_df.empty:
            df = pd.concat([cached_df, df]).sort_index().drop_duplicates()
        log.info("Successfully fetched historical data", rows=len(df))

        if use_cache:
            df.to_parquet(cache_path)
            log.info(f"Saved data to cache: {cache_path}")

        return df

    except IBConnectionError as e:
        log.error("IB connection failed", error=str(e))
        raise
    except NoDataError:
        log.error("No data error")
        raise
    except Exception as e:
        log.error("Unexpected error during async data fetch", error=str(e))
        raise DataFetchError(str(e)) from e

def fetch_historical_data(
    symbol: str,
    end_date: str = '',
    duration: str = '1 Y',
    bar_size: str = '1 day',
    ib_client: Optional[IB] = None,
    use_cache: bool = True,
    what_to_show: str = config.get('ib_api', {}).get('what_to_show', 'TRADES'),
    use_rth: bool = config.get('ib_api', {}).get('use_rth', True),
    format_date: int = config.get('ib_api', {}).get('format_date', 1)
) -> pd.DataFrame:
    """
    Fetches historical stock data for a single symbol from Interactive Brokers.

    Parameters:
        symbol: Stock ticker (e.g., 'AAPL').
        end_date: End date for data (format 'YYYYMMDD HH:MM:SS'; empty for current time).
        duration: Duration string (e.g., '1 Y', '6 M').
        bar_size: Granularity (e.g., '1 day', '5 mins').
        ib_client: Optional injected IB instance for testing.
        use_cache: If True, check cache before fetching and store results.
        what_to_show: Data type (e.g., 'TRADES', 'BID_ASK').
        use_rth: Restrict to regular trading hours.
        format_date: Date format for IB API (1 for string, 2 for Unix timestamp).

    Returns:
        Historical data as a pandas DataFrame indexed by timestamp.

    Raises:
        ValueError: If symbol or date validation fails.
        IBConnectionError: For connection-level issues.
        NoDataError: If IB returns no bars for the symbol.
        DataFetchError: For any other fetch or transformation errors.
    """
    validate_symbol(symbol)
    validate_date(end_date)
    log = logger.bind(symbol=symbol, duration=duration, end_date=end_date)
    log.info("Starting historical data fetch")

    cache_path = _get_cache_path(symbol) if use_cache else None
    cached_df = None

    if use_cache and cache_path.exists():
        cached_df = pd.read_parquet(cache_path)
        if not cached_df.empty:
            last_date = cached_df.index.max()
            log.info(f"Found cached data up to {last_date}")
            duration = "1 D"  # Fetch only new data
            end_date = last_date.strftime("%Y%m%d %H:%M:%S")

    try:
        if ib_client:
            ib = ib_client
            own_client = False
        else:
            from ib_api import ib_connection
            own_client = True

        if own_client:
            with ib_connection() as ib:
                bars = ib.reqHistoricalData(
                    Stock(symbol, 'SMART', 'USD'),
                    endDateTime=end_date,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=format_date
                )
        else:
            bars = ib.reqHistoricalData(
                Stock(symbol, 'SMART', 'USD'),
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=format_date
            )

        if not bars:
            if cached_df is not None and not cached_df.empty:
                log.info("No new data; returning cached data")
                return cached_df
            raise NoDataError(symbol)

        df = _bars_to_df(bars)
        if cached_df is not None and not cached_df.empty:
            df = pd.concat([cached_df, df]).sort_index().drop_duplicates()
        log.info("Successfully fetched historical data", rows=len(df))

        if use_cache:
            df.to_parquet(cache_path)
            log.info(f"Saved data to cache: {cache_path}")

        return df

    except IBConnectionError as e:
        log.error("IB connection failed", error=str(e))
        raise
    except NoDataError:
        log.error("No data error")
        raise
    except Exception as e:
        log.error("Unexpected error during data fetch", error=str(e))
        raise DataFetchError(str(e)) from e

async def fetch_multiple_historical_data(
    symbols: List[str],
    end_date: str = '',
    duration: str = '1 Y',
    bar_size: str = '1 day',
    use_cache: bool = True,
    what_to_show: str = config.get('ib_api', {}).get('what_to_show', 'TRADES'),
    use_rth: bool = config.get('ib_api', {}).get('use_rth', True),
    format_date: int = config.get('ib_api', {}).get('format_date', 1)
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple symbols concurrently, skipping failures.

    Parameters:
        symbols: List of tickers to fetch.
        end_date: End date string for all symbols.
        duration: Duration string for all symbols.
        bar_size: Bar size for all symbols.
        use_cache: If True, use and update cache.
        what_to_show: Data type (e.g., 'TRADES').
        use_rth: Restrict to regular trading hours.
        format_date: Date format for IB API.

    Returns:
        Mapping of symbol to its DataFrame. Symbols with errors are omitted.
    """
    from ib_api import ib_connection
    data: Dict[str, pd.DataFrame] = {}
    sem = Semaphore(5)  # Limit to 5 concurrent requests to respect IB rate limits

    async def fetch_all():
        with ib_connection() as ib:
            tasks = [
                _fetch_historical_async(
                    symbol, end_date, duration, bar_size, ib, sem,
                    what_to_show, use_rth, format_date, use_cache
                )
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for symbol, result in zip(symbols, results):
                if isinstance(result, pd.DataFrame):
                    data[symbol] = result
                elif isinstance(result, Exception):
                    logger.warning(f"Failed to fetch {symbol}", error=str(result))

    asyncio.run(fetch_all())
    return data