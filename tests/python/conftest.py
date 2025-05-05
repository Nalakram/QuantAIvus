from . import path_setup
import pytest
import pandas as pd
from unittest.mock import MagicMock, Mock
import srcPy.data.ib_data_collection as ibdc
from ib_insync import IB

@pytest.fixture(autouse=True)
def no_file_cache(monkeypatch, tmp_path):
    """
    Mock cache to raise FileNotFoundError by default. Override for cache-hit tests.
    """
    monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: tmp_path / f"{symbol}.parquet")
    monkeypatch.setattr(pd, "read_parquet", lambda path: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", Mock())  # Mock write by default

@pytest.fixture
def mock_ib(monkeypatch):
    """
    Mock IB client with realistic bars from create_mock_bars.
    """
    mock_ib_instance = MagicMock(spec=IB)
    bars = ibdc.create_mock_bars(5, start_date="2025-04-25")
    mock_ib_instance.reqHistoricalData.return_value = bars
    mock_ib_instance.reqHistoricalDataAsync.return_value = bars  # Return value directly, not coroutine
    monkeypatch.setattr("srcPy.data.ib_api.IB", Mock(return_value=mock_ib_instance))
    return mock_ib_instance

@pytest.fixture
def mock_ib_with_error(monkeypatch):
    """
    Mock IB client that raises IBConnectionError.
    """
    mock_ib_instance = MagicMock(spec=IB)
    mock_ib_instance.connect.side_effect = ConnectionError("connection lost")
    monkeypatch.setattr("srcPy.data.ib_api.IB", Mock(return_value=mock_ib_instance))
    return mock_ib_instance

@pytest.fixture
def mock_config():
    """
    Mock configuration for IB API settings.
    """
    return {
        'ib_api': {
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1,
            'what_to_show': 'TRADES',
            'use_rth': True,
            'format_date': 1
        }
    }