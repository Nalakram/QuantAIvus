import pytest
import pandas as pd
import numpy as np
from preprocessor import Preprocessor
from srcPy.utils.exceptions import DataValidationError

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Open": [1, 2, 3, 4],
        "High": [1.1, 2.1, 3.1, 4.1],
        "Low": [0.9, 1.9, 2.9, 3.9],
        "Close": [1, 2, 3, 4],
        "Volume": [100, 200, 300, 400],
        "date": pd.date_range("2023-01-01", periods=4)
    })

def test_validate_and_standardize(sample_df):
    p = Preprocessor()
    result = p._validate_and_standardize(sample_df.copy())
    assert all(c.islower() for c in result.columns)
    assert list(result.columns) == ["open", "high", "low", "close", "volume", "date"]

def test_validate_and_standardize_missing_columns():
    df = pd.DataFrame({"open": [1], "high": [2]})
    p = Preprocessor()
    with pytest.raises(DataValidationError, match="Missing required columns"):
        p._validate_and_standardize(df)

def test_fill_missing(sample_df):
    df = sample_df.copy()
    df.loc[1, "open"] = np.nan
    p = Preprocessor()
    result = p._fill_missing(df)
    assert not result["open"].isna().any()
    assert result["open"].iloc[1] == 1  # Forward-filled from 1

def test_add_indicators_rsi(mocker, sample_df):
    mocker.patch("pandas_ta.rsi", return_value=pd.Series([50, 60, 70, 80]))
    p = Preprocessor()
    df = p._validate_and_standardize(sample_df.copy())
    result = p._add_indicators(df)
    assert "rsi" in result.columns
    assert result["rsi"].isna().sum() == 0

def test_add_indicators_macd(mocker, sample_df):
    mocker.patch("pandas_ta.macd", return_value=pd.DataFrame({
        "MACD_12_26_9": [0.1, 0.2, 0.3, 0.4],
        "MACDs_12_26_9": [0.05, 0.15, 0.25, 0.35],
        "MACDh_12_26_9": [0.05, 0.05, 0.05, 0.05]
    }))
    p = Preprocessor()
    df = p._validate_and_standardize(sample_df.copy())
    result = p._add_indicators(df)
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_hist" in result.columns
    assert result["macd"].isna().sum() == 0

def test_process_custom_features_sentiment(sample_df):
    df = sample_df.copy()
    df["sentiment"] = [0.5, -0.5, 1.0, -1.5]
    p = Preprocessor()
    result = p._process_custom_features(df)
    assert result["sentiment"].between(-1.0, 1.0).all()
    assert result["sentiment"].iloc[3] == -1.0  # Clipped

def test_normalize_features(mocker, sample_df):
    mocker.patch("sklearn.preprocessing.StandardScaler.fit_transform", return_value=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]))
    p = Preprocessor()
    df = p._validate_and_standardize(sample_df.copy())
    result = p._normalize_features(df)
    assert result[["open", "close"]].values.tolist() == [[0, 0], [1, 1], [2, 2], [3, 3]]

def test_create_sequences(sample_df):
    p = Preprocessor()
    p.sequence_length = 2
    p.horizon = 1
    df = p._validate_and_standardize(sample_df.copy())
    data_array = df[["open", "close"]].values
    target_array = df["close"].values
    X_seq, y_seq = p._create_sequences(data_array, target_array)
    assert X_seq.shape == (2, 2, 2)  # 2 sequences, length 2, 2 features
    assert y_seq.shape == (2,)
    assert y_seq.tolist() == [3, 4]

def test_create_sequences_too_short():
    p = Preprocessor()
    p.sequence_length = 10
    p.horizon = 1
    df = pd.DataFrame({"open": [1, 2], "close": [1, 2]})
    data_array = df.values
    target_array = df["close"].values
    with pytest.raises(DataValidationError, match="Data length"):
        p._create_sequences(data_array, target_array)

def test_transform_single_ticker(sample_df, mocker):
    mocker.patch("pandas_ta.rsi", return_value=pd.Series([50, 60, 70, 80]))
    mocker.patch("sklearn.preprocessing.StandardScaler.fit_transform", return_value=np.array([[0, 0], [1, 1], [2, 2], [3, 3]]))
    p = Preprocessor()
    p.sequence_length = 2
    p.horizon = 1
    result = p.transform(sample_df.copy())
    X, y = result
    assert X.shape == (2, 2, 7)  # 2 sequences, length 2, features including rsi
    assert y.shape == (2,)
    assert y.tolist() == [3, 4]

def test_transform_multi_ticker(sample_multi_ticker, mocker):
    mocker.patch("pandas_ta.rsi", return_value=pd.Series([50, 60]))
    mocker.patch("sklearn.preprocessing.StandardScaler.fit_transform", return_value=np.array([[0, 0], [1, 1]]))
    p = Preprocessor()
    p.sequence_length = 1
    p.horizon = 1
    result = p.transform(pd.read_csv(sample_multi_ticker))
    assert isinstance(result, dict)
    assert "A" in result and "B" in result
    assert result["A"][0].shape == (1, 1, 7)
    assert result["B"][0].shape == (1, 1, 7)