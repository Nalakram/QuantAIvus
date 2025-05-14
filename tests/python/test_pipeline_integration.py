import pytest
import pandas as pd
import asyncio
from data_loader import CSVLoader, AlpacaStreamLoader
from data_cleaning import DataCleaner, StreamingCleanerPipeline, MissingImputer
from preprocessor import Preprocessor

def test_end_to_end_pipeline(sample_csv, mocker):
    mocker.patch("mlflow.start_run")
    mocker.patch("mlflow.log_metric")
    mocker.patch("mlflow.log_param")
    mocker.patch("pandas_ta.rsi", return_value=pd.Series([50, 60]))
    loader_conf = type("C", (), {"path": sample_csv, "chunksize": 1000, "use_dask": False})
    df = CSVLoader(loader_conf).load_data()
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df.assign(ticker="T"))
    preprocessor = Preprocessor()
    preprocessor.sequence_length = 1
    preprocessor.horizon = 1
    result = preprocessor.transform(df_clean)
    assert isinstance(result, dict)
    assert "T" in result
    X, y = result["T"]
    assert X.shape[0] == 2
    assert X.shape[1] == 1
    assert X.shape[2] >= 5  # At least OHLCV, possibly rsi, macd, etc.
    assert y.shape == (2,)

def test_end_to_end_multi_ticker(sample_multi_ticker, mocker):
    mocker.patch("mlflow.start_run")
    mocker.patch("mlflow.log_metric")
    mocker.patch("mlflow.log_param")
    mocker.patch("pandas_ta.rsi", return_value=pd.Series([50, 60]))
    loader_conf = type("C", (), {"path": sample_multi_ticker, "chunksize": 1000, "use_dask": False})
    df = CSVLoader(loader_conf).load_data()
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df)
    preprocessor = Preprocessor()
    preprocessor.sequence_length = 1
    preprocessor.horizon = 1
    result = preprocessor.transform(df_clean)
    assert isinstance(result, dict)
    assert "A" in result and "B" in result
    assert result["A"][0].shape == (1, 1, 7)
    assert result["B"][0].shape == (1, 1, 7)

def test_distributed_pipeline(sample_csv, mocker):
    mocker.patch("mlflow.start_run")
    mocker.patch("mlflow.log_metric")
    mocker.patch("mlflow.log_param")
    mocker.patch("data_cleaning.config.distributed_processing.min_rows_for_distributed", 1)
    loader_conf = type("C", (), {"path": sample_csv, "chunksize": 1000, "use_dask": True})
    df = CSVLoader(loader_conf).load_data()
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df)
    assert isinstance(df_clean, pd.DataFrame)
    assert len(df_clean) == len(df)

@pytest.mark.asyncio
async def test_streaming_pipeline_integration(mocker):
    mocker.patch("mlflow.start_run")
    mocker.patch("mlflow.log_metric")
    mocker.patch("mlflow.log_param")
    loader = AlpacaStreamLoader(config.data_source.real_time_market_data.alpaca)
    mock_ws = AsyncMock(recv=AsyncMock(side_effect=[
        '{"trade":"data","close":1,"open":1,"high":1,"low":1,"volume":100}',
        '{"trade":"data","close":2,"open":np.nan,"high":2,"low":2,"volume":200}',
        None
    ]))
    mocker.patch("websockets.connect", return_value=mock_ws)
    steps = [MissingImputer("forward_fill", {"backward_fill": False})]
    pipeline = StreamingCleanerPipeline(steps, buffer_size=2)
    results = []
    async for cleaned in pipeline.process_stream(loader.stream_data()):
        results.append(cleaned)
        break
    assert len(results) == 1
    assert isinstance(results[0], pd.DataFrame)
    assert "close" in results[0].columns
    assert not results[0]["open"].isna().any()
    assert "rsi" in results[0].columns
    assert "macd" in results[0].columns