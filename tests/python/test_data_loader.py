import pytest
import pandas as pd
import asyncio
from unittest.mock import AsyncMock
from aioresponses import aioresponses
from data_loader import CSVLoader, TwitterLoader, AlpacaStreamLoader, ESGLoader, FREDLoader, BloombergLoader, WeatherLoader, InfluxDBLoader, build_loader
from srcPy.utils.exceptions import DataFetchError, IBConnectionError, NoDataError

def test_csv_loader_happy_path(sample_csv, mocker):
    mock_log_metric = mocker.patch("mlflow.log_metric")
    mock_log_param = mocker.patch("mlflow.log_param")
    conf = type("C", (), {"path": sample_csv, "chunksize": 1000, "use_dask": False})
    loader = CSVLoader(conf)
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    mock_log_metric.assert_called_with("csv_rows_loaded", 2)
    mock_log_param.assert_called_with("csv_schema", ["open", "high", "low", "close", "volume"])

def test_csv_loader_invalid_path(tmp_path):
    conf = type("C", (), {"path": str(tmp_path / "nofile.csv"), "chunksize": 1000, "use_dask": False})
    loader = CSVLoader(conf)
    with pytest.raises(FileNotFoundError):
        loader.load_data()

@pytest.mark.asyncio
async def test_csv_loader_stream(sample_csv):
    conf = type("C", (), {"path": sample_csv, "chunksize": 1, "use_dask": False})
    loader = CSVLoader(conf)
    chunks = []
    async for chunk in loader.stream_data():
        chunks.append(chunk)
    assert len(chunks) == 2
    assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)
    assert chunks[0].shape == (1, 5)

@pytest.mark.asyncio
async def test_twitter_loader_stream(mocker):
    loader = TwitterLoader(config.alternative_data.twitter)
    mock_response = mocker.patch("aiohttp.ClientSession.get")
    mock_response.return_value.__aenter__.return_value.content = AsyncMock(
        return_value=[b'{"text":"test tweet"}']
    )
    async for data in loader.stream_data():
        assert isinstance(data, bytes)
        assert b"test tweet" in data
        break

@pytest.mark.asyncio
async def test_twitter_loader_stream_failure():
    loader = TwitterLoader(config.alternative_data.twitter)
    with aioresponses() as m:
        m.get(
            loader.base_url + loader.endpoints['filtered_stream'],
            status=429,
            headers={"Retry-After": "60"}
        )
        with pytest.raises(DataFetchError):
            async for _ in loader.stream_data():
                pass

@pytest.mark.asyncio
async def test_twitter_loader_no_data(mocker):
    mocker.patch("data_loader.APIDataLoader.load_data", return_value=pd.DataFrame())
    loader = TwitterLoader(config.alternative_data.twitter)
    with pytest.raises(NoDataError, match="No data returned"):
        await loader.load_data(query="market")

@pytest.mark.asyncio
async def test_alpaca_stream_loader(mocker):
    loader = AlpacaStreamLoader(config.data_source.real_time_market_data.alpaca)
    mock_ws = AsyncMock(recv=AsyncMock(return_value='{"trade":"data","close":1}'))
    mocker.patch("websockets.connect", return_value=mock_ws)
    async for msg in loader.stream_data():
        assert isinstance(msg, str)
        assert "trade" in msg
        break

@pytest.mark.asyncio
async def test_alpaca_stream_connection_error(mocker):
    loader = AlpacaStreamLoader(config.data_source.real_time_market_data.alpaca)
    mocker.patch("websockets.connect", side_effect=Exception("Connection failed"))
    with pytest.raises(IBConnectionError, match="WebSocket connection failed"):
        async for _ in loader.stream_data():
            pass

@pytest.mark.asyncio
async def test_esg_loader(mocker):
    loader = ESGLoader(config.alternative_data.esg)
    mock_response = AsyncMock(return_value=[{"id": "AAPL", "score": 0.8}])
    mocker.patch("data_loader.APIDataLoader._request", mock_response)
    df = await loader.load_data(["AAPL"])
    assert isinstance(df, pd.DataFrame)
    assert "score" in df.columns
    assert df["score"].iloc[0] == 0.8

@pytest.mark.asyncio
async def test_fred_loader(mocker):
    loader = FREDLoader(config.alternative_data.fred)
    mock_response = AsyncMock(return_value=[{"series_id": "GDP", "value": 1000}])
    mocker.patch("data_loader.APIDataLoader._request", mock_response)
    df = await loader.load_data(["GDP"])
    assert isinstance(df, pd.DataFrame)
    assert "value" in df.columns
    assert df["value"].iloc[0] == 1000

@pytest.mark.asyncio
async def test_bloomberg_loader(mocker):
    loader = BloombergLoader(config.alternative_data.bloomberg)
    mock_response = AsyncMock(return_value=[{"topic": "markets", "news": "update"}])
    mocker.patch("data_loader.APIDataLoader._request", mock_response)
    df = await loader.load_data(["markets"])
    assert isinstance(df, pd.DataFrame)
    assert "news" in df.columns
    assert df["news"].iloc[0] == "update"

@pytest.mark.asyncio
async def test_weather_loader(mocker):
    loader = WeatherLoader(config.alternative_data.weather)
    mock_response = AsyncMock(return_value=[{"city": "New York", "temp": 20}])
    mocker.patch("data_loader.APIDataLoader._request", mock_response)
    df = await loader.load_data(["New York"])
    assert isinstance(df, pd.DataFrame)
    assert "temp" in df.columns
    assert df["temp"].iloc[0] == 20

def test_influxdb_loader_config_validation():
    conf = type("C", (), {"host": None, "port": 8086, "token": "token", "org": "org", "bucket": "bucket", "query": "query"})
    with pytest.raises(ValueError, match="Missing InfluxDB config: host"):
        InfluxDBLoader(conf)

@pytest.mark.parametrize("loader_type,cls", [
    ("csv", CSVLoader),
    ("influxdb", InfluxDBLoader),
    ("twitter", TwitterLoader),
    ("alpaca_stream", AlpacaStreamLoader),
    ("esg", ESGLoader),
    ("fred", FREDLoader),
    ("bloomberg", BloombergLoader),
    ("weather", WeatherLoader),
])
def test_build_loader(loader_type, cls, mocker):
    mocker.patch("data_loader.get_config", return_value=type(
        "C", (), {
            "data_source": type("C", (), {"type": loader_type, loader_type: {}}),
            "alternative_data": type("C", (), {loader_type: {}})
        }
    ))
    loader = build_loader(loader_type)
    assert isinstance(loader, cls)

def test_build_loader_unsupported():
    with pytest.raises(ValueError, match="Unsupported loader type: invalid"):
        build_loader("invalid")