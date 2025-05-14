import asyncio

import numpy as np
import pandas as pd
import pytest
from data_cleaning import (
    AnomalyDetector,
    CalendarFeatures,
    CleanerPipeline,
    IncrementalMACDStep,
    IncrementalRSIStep,
    MissingImputer,
    OutlierHandler,
    SentimentExtractor,
    StreamingAnomalyStep,
    StreamingCleanerPipeline,
    ValidationStep,
)

from srcPy.utils.exceptions import DataValidationError


def test_missing_imputer_forward_fill():
    df = pd.DataFrame({"A": [1, None, None, 4], "B": [None, 2, 3, None]})
    step = MissingImputer("forward_fill", {"backward_fill": True})
    result = step.apply(df.copy())
    assert result["A"].tolist() == [1, 1, 1, 4]
    assert result["B"].tolist() == [2, 2, 3, 3]

@pytest.mark.parametrize("method,expected", [
    ("forward_fill", [1, 1, 1, 4]),
    ("interpolate", [1, 2, 3, 4])
])
def test_missing_imputer_methods(method, expected):
    df = pd.DataFrame({"A": [1, None, None, 4]})
    step = MissingImputer(method, {})
    result = step.apply(df.copy())
    assert result["A"].tolist() == expected

def test_outlier_handler_zscore():
    df = pd.DataFrame({"x": [1, 2, 100, 3, 4]})
    step = OutlierHandler("zscore", {"threshold": 2})
    result = step.apply(df.copy())
    assert result["x"].max() < 100
    assert not result["x"].isna().all()

def test_outlier_handler_empty_df():
    df = pd.DataFrame({"x": []})
    step = OutlierHandler("zscore", {"threshold": 2})
    result = step.apply(df.copy())
    assert result.empty

def test_sentiment_extractor_no_text():
    df = pd.DataFrame({"close": [1, 2, 3]})
    step = SentimentExtractor(type("C", (), {"enabled": True}))
    result = step.apply(df.copy())
    assert "sentiment" in result.columns
    assert result["sentiment"].eq(0.0).all()

def test_anomaly_detector():
    df = pd.DataFrame({"close": [1, 1, 1, 100, 1]})
    step = AnomalyDetector(type("C", (), {"enabled": True, "method": "isolation_forest", "params": {"contamination": 0.2}}))
    result = step.apply(df.copy())
    assert len(result) < len(df)

def test_incremental_rsi():
    df = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
    step = IncrementalRSIStep(window=2)
    result = step.apply(df.copy())
    assert "rsi" in result.columns
    assert result["rsi"].isna().sum() == 2

def test_incremental_macd():
    df = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
    step = IncrementalMACDStep(fast=12, slow=26, signal=9)
    result = step.apply(df.copy())
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert result["macd"].isna().sum() == 0

def test_calendar_features():
    df = pd.DataFrame(index=pd.date_range("2023-01-01", periods=3))
    step = CalendarFeatures(type("C", (), {"day_of_week": True, "is_holiday": True}))
    result = step.apply(df.copy())
    assert "day_of_week" in result.columns
    assert "is_holiday" in result.columns

def test_streaming_anomaly_step():
    df = pd.DataFrame({"close": [1, 1, 100, 1]})
    step = StreamingAnomalyStep(contamination=0.25, refit_every=2)
    result = step.apply(df.copy())
    assert len(result) < len(df)

def test_validation_step():
    df = pd.DataFrame({"open": [1], "high": [2]})  # Missing low, close, volume
    step = ValidationStep()
    with pytest.raises(DataValidationError, match="Missing required columns"):
        step.apply(df.copy())

def test_validation_step_duplicates():
    df = pd.DataFrame({
        "open": [1, 2],
        "high": [2, 3],
        "low": [0.5, 1],
        "close": [1.5, 2.5],
        "volume": [100, 200]
    }, index=[pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")])
    step = ValidationStep()
    with pytest.raises(DataValidationError, match="Duplicate timestamps"):
        step.apply(df.copy())

def test_pipeline_run(sample_csv, mocker):
    mocker.patch("mlflow.start_run")
    mocker.patch("mlflow.log_param")
    df = pd.read_csv(sample_csv)
    steps = [
        MissingImputer("forward_fill", {"backward_fill": False}),
        OutlierHandler("zscore", {"threshold": 3})
    ]
    pipeline = CleanerPipeline(steps)
    cleaned = pipeline.run(df.copy(), distributed=False)
    assert not cleaned.isnull().values.any()
    assert cleaned.shape[1] == df.shape[1]

def test_pipeline_distributed(sample_csv, mocker):
    mocker.patch("mlflow.start_run")
    mocker.patch("mlflow.log_param")
    mocker.patch("data_cleaning.config.distributed_processing.min_rows_for_distributed", 1)
    df = pd.read_csv(sample_csv)
    steps = [MissingImputer("forward_fill", {"backward_fill": False})]
    pipeline = CleanerPipeline(steps)
    cleaned = pipeline.run(df.copy(), distributed=True)
    assert isinstance(cleaned, pd.DataFrame)
    assert len(cleaned) == len(df)

async def fake_stream():
    for i in range(3):
        yield {"close": i, "open": None if i % 2 == 0 else i}
        await asyncio.sleep(0)

@pytest.mark.asyncio
async def test_streaming_cleaner_pipeline(mocker):
    mocker.patch("mlflow.log_metric")
    steps = [MissingImputer("forward_fill", {"backward_fill": False})]
    pipeline = StreamingCleanerPipeline(steps, buffer_size=2)
    results = []
    async for cleaned in pipeline.process_stream(fake_stream()):
        results.append(cleaned)
    assert len(results) == 2
    for df in results:
        assert not df["open"].isnull().any()
        assert "rsi" in df.columns
        assert "macd" in df.columns

def test_mlflow_metrics_logging(mocker):
    mock_log_metric = mocker.patch("mlflow.log_metric")
    df = pd.DataFrame({"A": [None, 2, None, 4]})
    step = MissingImputer("forward_fill", {"backward_fill": False})
    step.apply(df.copy())
    mock_log_metric.assert_called_with("missing_imputed", 2)

def test_cleaner_empty_dataframe():
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    steps = [MissingImputer("forward_fill", {"backward_fill": False})]
    pipeline = CleanerPipeline(steps)
    cleaned = pipeline.run(df.copy(), distributed=False)
    assert cleaned.empty
    assert list(cleaned.columns) == ["open", "high", "low", "close", "volume"]
