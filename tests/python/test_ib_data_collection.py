import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
from ib_insync import IB, BarData, Stock
from python.data.ib_data_collection import (
    fetch_historical_data,
    fetch_multiple_historical_data,
    _bars_to_df,
    create_mock_bars,
    NoDataError,
    DataFetchError,
)
from python.utils.exceptions import IBConnectionError

# Mock configuration
@pytest.fixture
def mock_config():
    return {
        'ib_api': {
            'what_to_show': 'TRADES',
            'use_rth': True,
            'format_date': 1
        }
    }

# Mock IB client
@pytest.fixture
def mock_ib():
    ib = Mock(spec=IB)
    ib.reqHistoricalData.return_value = create_mock_bars(5)
    ib.reqHistoricalDataAsync = Mock(return_value=create_mock_bars(5))
    return ib

# Test create_mock_bars
def test_create_mock_bars():
    bars = create_mock_bars(3, start_date="2025-01-01")
    assert len(bars) == 3
    assert isinstance(bars[0], BarData)
    assert bars[0].date == "20250101 00:00:00"
    assert bars[0].open == 100.0
    assert bars[2].close == 102.5

# Test _bars_to_df
def test_bars_to_df():
    bars = create_mock_bars(3)
    df = _bars_to_df(bars)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume', 'barCount', 'average']
    assert df.index.name == 'date'
    assert not df.isna().any().any()

def test_bars_to_df_empty():
    with pytest.raises(DataFetchError, match="Empty DataFrame after conversion"):
        _bars_to_df([])

def test_bars_to_df_missing_columns():
    bars = create_mock_bars(1)
    bars[0].open = None  # Simulate missing data
    with pytest.raises(DataFetchError, match="Missing expected columns"):
        _bars_to_df(bars)

# Test fetch_historical_data
@patch('python.data.ib_data_collection.pd.read_parquet')
@patch('python.data.ib_data_collection.Path')
@patch('python.data.ib_data_collection.config', new_callable=lambda: mock_config)
def test_fetch_historical_data_success(mock_config, mock_path, mock_read_parquet, mock_ib):
    mock_read_parquet.return_value = pd.DataFrame()  # Empty cache
    mock_path.return_value.exists.return_value = False
    df = fetch_historical_data(
        symbol='AAPL',
        end_date='20250430 16:00:00',
        duration='1 Y',
        bar_size='1 day',
        ib_client=mock_ib
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    mock_ib.reqHistoricalData.assert_called_once()

def test_fetch_historical_data_no_data(mock_ib):
    mock_ib.reqHistoricalData.return_value = []
    with pytest.raises(NoDataError, match="No historical data returned for AAPL"):
        fetch_historical_data(
            symbol='AAPL',
            end_date='20250430 16:00:00',
            duration='1 Y',
            bar_size='1 day',
            ib_client=mock_ib,
            use_cache=False
        )

@patch('python.data.ib_data_collection.validate_symbol')
def test_fetch_historical_data_invalid_symbol(mock_validate_symbol, mock_ib):
    mock_validate_symbol.side_effect = ValueError("Invalid symbol")
    with pytest.raises(ValueError, match="Invalid symbol"):
        fetch_historical_data(
            symbol='INVALID',
            end_date='20250430 16:00:00',
            duration='1 Y',
            bar_size='1 day',
            ib_client=mock_ib
        )

# Test fetch_multiple_historical_data
@patch('python.data.ib_data_collection._fetch_historical_async')
@patch('python.data.ib_data_collection.ib_connection')
def test_fetch_multiple_historical_data(mock_ib_connection, mock_fetch_async):
    mock_ib = Mock(spec=IB)
    mock_ib_connection.return_value.__enter__.return_value = mock_ib
    mock_fetch_async.side_effect = [
        pd.DataFrame({'open': [100], 'high': [101], 'low': [99], 'close': [100.5], 'volume': [1000]}, index=[pd.to_datetime('2025-01-01')]),
        Exception("Failed to fetch")
    ]
    result = fetch_multiple_historical_data(
        symbols=['AAPL', 'MSFT'],
        end_date='20250430 16:00:00',
        duration='1 Y',
        bar_size='1 day'
    )
    assert 'AAPL' in result
    assert 'MSFT' not in result
    assert isinstance(result['AAPL'], pd.DataFrame)