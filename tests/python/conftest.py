from unittest.mock import MagicMock, Mock

import mlflow
import pandas as pd
import pytest
from ib_insync import IB

import srcPy.data.ib_data_collection as ibdc

from . import path_setup


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

class DummyMLflowRun:
    def __init__(self):
        self.metrics = {}
        self.params = {}
    def log_metric(self, key, value):
        self.metrics.setdefault(key, []).append(value)
    def log_param(self, key, value):
        self.params[key] = value

@pytest.fixture(autouse=True)
def mlflow_mock(monkeypatch):
    """Auto-used fixture to mock mlflow logging."""
    dummy = DummyMLflowRun()
    monkeypatch.setattr(mlflow, "log_metric", dummy.log_metric)
    monkeypatch.setattr(mlflow, "log_param", dummy.log_param)
    # If start_run/end_run are used, also patch them to no-op
    monkeypatch.setattr(mlflow, "start_run", lambda **kwargs: None)
    monkeypatch.setattr(mlflow, "end_run", lambda **kwargs: None)
    return dummy

@pytest.fixture(autouse=True, scope="session")
def env_api_keys(monkeypatch):
    monkeypatch.setenv("INFLUXDB_TOKEN", "fake_token")
    monkeypatch.setenv("IB_API_KEY", "fake_ib_key")
    monkeypatch.setenv("ALPACA_KEY", "fake_alpaca")
    monkeypatch.setenv("ALPACA_SECRET", "fake_secret")
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "fake_twitter")
    monkeypatch.setenv("ESG_API_KEY", "fake_esg")
    monkeypatch.setenv("FRED_API_KEY", "fake_fred")
    monkeypatch.setenv("BBG_API_KEY", "fake_bbg")
    monkeypatch.setenv("WEATHER_API_KEY", "fake_weather")

@pytest.fixture(autouse=True)
def mock_mlflow(mocker):
    mocker.patch("mlflow.log_metric")
    mocker.patch("mlflow.log_param")
    mocker.patch("mlflow.start_run")

@pytest.fixture(autouse=True)
def mock_pandas_ta(mocker):
    mocker.patch("pandas_ta.rsi", return_value=pd.Series([50, 60, 70, 80]))
    mocker.patch("pandas_ta.macd", return_value=pd.DataFrame({
        "MACD_12_26_9": [0.1, 0.2, 0.3, 0.4],
        "MACDs_12_26_9": [0.05, 0.15, 0.25, 0.35],
        "MACDh_12_26_9": [0.05, 0.05, 0.05, 0.05]
    }))
