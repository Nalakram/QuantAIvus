import asyncio
import json
from pathlib import Path

import pandas as pd
import pyarrow
import pytest

print(pyarrow.__version__)
df = pd.DataFrame({"a": [1, 2]})
df.to_parquet("test.parquet", engine="pyarrow")
print(pd.read_parquet("test.parquet"))
ORIGINAL_READ_PARQUET = pd.read_parquet
ORIGINAL_TO_PARQUET = pd.DataFrame.to_parquet
from unittest.mock import Mock

import srcPy.data.ib_data_collection as ibdc
from srcPy.utils.exceptions import DataFetchError, IBConnectionError, NoDataError


class TestBarsToDf:
    @pytest.mark.parametrize("mutate, error_msg", [
        (lambda bars: bars.clear(), "Empty DataFrame"),
        (lambda bars: bars.__setitem__(slice(None), [
            Mock(date=b.date, high=b.high, low=b.low, close=b.close,
                 volume=b.volume, average=b.average, barCount=b.barCount,
                 open=None)
            for b in bars
        ]),
         r"Missing BarData fields:.*open.*"),
    ])
    def test_bars_to_df_errors(self, mutate, error_msg):
        bars = ibdc.create_mock_bars(3)
        mutate(bars)
        with pytest.raises(DataFetchError, match=error_msg):
            ibdc._bars_to_df(bars)

    def test_bars_to_df_success(self):
        bars = ibdc.create_mock_bars(3)
        df = ibdc._bars_to_df(bars)
        assert list(df.columns) == ["open", "high", "low", "close", "volume", "average", "barCount"]
        assert len(df) == 3

    def test_bars_to_df_duplicate_index(self):
        bars = ibdc.create_mock_bars(2, start_date="2025-01-01")
        bars.append(bars[0])
        df = ibdc._bars_to_df(bars)
        assert len(df) == 2
        assert df.index.duplicated().sum() == 0

    def test_bars_to_df_nan_handling(self, caplog):
        bars = ibdc.create_mock_bars(2)
        bars[1].open = float('nan')
        with caplog.at_level("WARNING"):
            df = ibdc._bars_to_df(bars)
        for record in caplog.records:
            try:
                log_data = json.loads(record.message)
                if log_data.get("event") == "Missing values detected in DataFrame":
                    assert log_data.get("action") == "filling with forward fill"
                    break
            except json.JSONDecodeError:
                continue
        else:
            assert False, "Expected warning log not found"
        assert not df["open"].isna().any()


class TestFetchHistoricalData:
    def test_cache_hit(self, tmp_path, mock_ib, monkeypatch):
        df_cached = pd.DataFrame(
            {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10], 'average': [1.5], 'barCount': [1]},
            index=[pd.Timestamp("2025-01-01", tz="UTC")]
        )
        cache_file = tmp_path / "AAPL.parquet"
        df_cached.to_parquet(cache_file)
        #  Point _get_cache_path at our “fake” cache path
        monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
        # Pretend the file really exists
        monkeypatch.setattr(Path, "exists", lambda self: True)

        # Return our df whenever read_parquet is called
        monkeypatch.setattr(pd, "read_parquet", lambda path, **kw: df_cached)
        df = ibdc.fetch_historical_data(
            symbol="AAPL",
            end_date="20241231 16:00:00",
            ib_client=mock_ib,
            use_cache=True
        )

        mock_ib.reqHistoricalData.assert_not_called()
        pd.testing.assert_frame_equal(df, df_cached)


def test_cache_update(tmp_path, mock_ib, monkeypatch):
    # Set up a non-existent cache path
    cache_file = tmp_path / "AAPL.parquet"
    monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)

    # Make read_parquet raise FileNotFoundError before the fetch
    monkeypatch.setattr(pd, "read_parquet", lambda path, **kw: (_ for _ in ()).throw(FileNotFoundError()))

    # Restore the real to_parquet so the file actually lands on disk
    monkeypatch.setattr(pd.DataFrame, "to_parquet", ORIGINAL_TO_PARQUET)

    # Stub IB to return 2 new bars
    new_bars = ibdc.create_mock_bars(2, start_date="2025-01-02")
    mock_ib.reqHistoricalData.return_value = new_bars
    df = ibdc.fetch_historical_data(
        symbol="AAPL",
        end_date="20250103 16:00:00",
        ib_client=mock_ib,
        use_cache=True
    )
    assert len(df) == 2
    assert df.index[-1] == pd.Timestamp("2025-01-03", tz="UTC")

    # Restore read_parquet to the original pandas implementation
    monkeypatch.setattr(pd, "read_parquet", ORIGINAL_READ_PARQUET)

    # Read the cache file and verify it matches the fetched data
    updated_cache = pd.read_parquet(cache_file)
    pd.testing.assert_frame_equal(df, updated_cache)


def test_cache_empty_file(tmp_path, mock_ib, monkeypatch):
    cache_file = tmp_path / "AAPL.parquet"
    cache_file.write_bytes(b"")
    monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
    mock_ib.reqHistoricalData.return_value = ibdc.create_mock_bars(1)
    df = ibdc.fetch_historical_data(
        symbol="AAPL",
        end_date="20250430 16:00:00",
        ib_client=mock_ib,
        use_cache=True
    )
    assert not df.empty
    mock_ib.reqHistoricalData.assert_called_once()


def test_no_cache_fetch(mock_ib):
    mock_ib.reqHistoricalData.return_value = ibdc.create_mock_bars(1)
    df = ibdc.fetch_historical_data(
        symbol="AAPL",
        end_date="20250430 16:00:00",
        ib_client=mock_ib,
        use_cache=False
    )
    mock_ib.reqHistoricalData.assert_called_once()
    assert not df.empty


def test_fetch_historical_data_no_data_with_cache(tmp_path, mock_ib, monkeypatch):
    df_cached = pd.DataFrame(
        {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10], 'average': [1.5], 'barCount': [1]},
        index=[pd.Timestamp("2025-01-01", tz="UTC")]
    )
    cache_file = tmp_path / "AAPL.parquet"
    df_cached.to_parquet(cache_file)
    monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    # ensure read_parquet returns our cached DF
    monkeypatch.setattr(pd, "read_parquet", lambda path, **kw: df_cached)
    mock_ib.reqHistoricalData.return_value = []

    df = ibdc.fetch_historical_data(
        symbol="AAPL",
        end_date="20250101 16:00:00",
        ib_client=mock_ib,
        use_cache=True
    )
    pd.testing.assert_frame_equal(df, df_cached)


def test_invalid_date_raises(mock_ib):
    with pytest.raises(ValueError):
        ibdc.fetch_historical_data(
            symbol="AAPL",
            end_date="invalid-date",
            ib_client=mock_ib,
            use_cache=False
        )


def test_invalid_symbol_raises(mock_ib, monkeypatch):
    monkeypatch.setattr(ibdc, "validate_symbol",
                        lambda x: (_ for _ in ()).throw(ValueError("Invalid symbol")))
    monkeypatch.setattr(ibdc, "validate_date",
                        lambda x: (_ for _ in ()).throw(ValueError("Bad date")))
    with pytest.raises(ValueError, match=r"Invalid symbol"):
        ibdc.fetch_historical_data(
            symbol="INVALID",
            end_date="20250430 16:00:00",
            ib_client=mock_ib,
            use_cache=False
        )


def test_missing_config(mock_ib, monkeypatch):
    monkeypatch.setattr("srcPy.utils.config.config", {})
    df = ibdc.fetch_historical_data(symbol="AAPL", end_date="20250430 16:00:00", ib_client=mock_ib, use_cache=False)
    assert not df.empty


def test_ib_error_bubbles(mock_ib_with_error, monkeypatch):
    monkeypatch.setattr("ib_insync.util.df", lambda x: (_ for _ in ()).throw(IBConnectionError("connection lost")))
    with pytest.raises(IBConnectionError, match="connection lost"):
        ibdc.fetch_historical_data(
            symbol="AAPL",
            end_date="20250430 16:00:00",
            ib_client=mock_ib_with_error,
            use_cache=False
        )



def test_no_data_no_cache(mock_ib):
    mock_ib.reqHistoricalData.return_value = []
    print("Testing NoDataError for test_no_data_no_cache")
    with pytest.raises(NoDataError, match="No historical data returned for AAPL"):
        ibdc.fetch_historical_data(
            symbol="AAPL",
            end_date="20250430 16:00:00",
            ib_client=mock_ib,
            use_cache=False
        )
    print("NoDataError successfully caught")


def test_invalid_cache_columns(tmp_path, mock_ib, monkeypatch):
    df_cached = pd.DataFrame({'invalid': [1]}, index=[pd.Timestamp("2025-01-01", tz="UTC")])
    cache_file = tmp_path / "AAPL.parquet"
    df_cached.to_parquet(cache_file)
    monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
    mock_ib.reqHistoricalData.return_value = ibdc.create_mock_bars(1)
    df = ibdc.fetch_historical_data(
        symbol="AAPL",
        end_date="20250102 16:00:00",
        ib_client=mock_ib,
        use_cache=True
    )
    assert not df.empty


from srcPy.data.ib_data_collection import NoDataError


class TestAsyncHelpers:
    @pytest.mark.asyncio
    async def test_fetch_historical_async_no_data(self, mock_ib):
        sem = asyncio.Semaphore(1)
        mock_ib.reqHistoricalDataAsync.return_value = []
        print("Testing NoDataError for test_fetch_historical_async_no_data")
        with pytest.raises(NoDataError, match="No historical data returned for AAPL"):
            await ibdc._fetch_historical_async(
                "AAPL", "20250101 00:00:00", "1 D", "1 day",
                mock_ib, sem, "TRADES", True, 1, False
            )
        print("NoDataError successfully caught")

    @pytest.mark.asyncio
    async def test_fetch_historical_async_caches_and_returns(self, tmp_path, monkeypatch, mock_ib):
        cache_file = tmp_path / "AAPL.parquet"
        monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
        monkeypatch.setattr(Path, "exists", lambda self: True)
        sem = asyncio.Semaphore(1)
        bars = ibdc.create_mock_bars(5, start_date="2025-01-01")
        mock_ib.reqHistoricalDataAsync.return_value = bars

        # restore real to_parquet so the async helper actually writes a file
        monkeypatch.setattr(pd.DataFrame, "to_parquet", ORIGINAL_TO_PARQUET)
        df = await ibdc._fetch_historical_async(
            "AAPL", "20250105 00:00:00", "1 D", "1 day",
            mock_ib, sem, "TRADES", True, 1, True
        )
        assert cache_file.exists()
        pd.testing.assert_frame_equal(df, ibdc._bars_to_df(bars))

    @pytest.mark.asyncio
    async def test_fetch_historical_async_cache_update(self, tmp_path, monkeypatch, mock_ib, caplog):
        # 1. Create a cached DataFrame
        df_cached = pd.DataFrame(
            {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10], 'average': [1.5], 'barCount': [1]},
            index=[pd.Timestamp("2025-01-01", tz="UTC")]
        )
        cache_file = tmp_path / "AAPL.parquet"

        # 2. Mock cache path and parquet reading
        monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
        monkeypatch.setattr(Path, "exists", lambda self: True)

        def fake_read_parquet(path):
            return df_cached.copy()

        monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

        # 3. Mock IB to return new bars
        new_bars = ibdc.create_mock_bars(2, start_date="2025-01-02")
        mock_ib.reqHistoricalDataAsync.return_value = new_bars

        # 4. Mock DataFrame.to_parquet to avoid disk I/O
        saved = {}

        def fake_to_parquet(self, path, engine):
            saved["path"] = path
            saved["df"] = self.copy()

        monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

        # 5. Capture logs
        caplog.set_level("INFO")

        # 6. Call the helper
        df = await ibdc._fetch_historical_async(
            "AAPL", "20250103 16:00:00", "1 D", "1 day",
            mock_ib, asyncio.Semaphore(1), "TRADES",
            use_rth=True, format_date=1, use_cache=True
        )

        # 7. Verify results
        assert len(df) == 3  # 1 cached + 2 new
        assert df.index[-1] == pd.Timestamp("2025-01-03", tz="UTC")

        # 8. Verify cache save
        assert saved["path"] == cache_file
        pd.testing.assert_frame_equal(saved["df"], df)

        # 9. Verify logs
        log_events = [json.loads(rec.message).get("event") for rec in caplog.records]
        assert "Found cached data" in log_events
        assert "Successfully fetched historical data" in log_events
        assert "Saved data to cache" in log_events

    @pytest.mark.asyncio
    async def test_fetch_historical_async_timeout(self, mock_ib):
        mock_ib.reqHistoricalDataAsync.side_effect = asyncio.TimeoutError("Request timed out")
        sem = asyncio.Semaphore(1)
        with pytest.raises(DataFetchError, match="Request timed out"):
            await ibdc._fetch_historical_async(
                "AAPL", "20250101 16:00:00", "1 D", "1 day",
                mock_ib, sem, "TRADES", True, 1, False
            )


class TestFetchMultiple:
    def test_fetch_multiple_historical_data(self, monkeypatch, mock_ib):
        from srcPy.data import ib_api
        monkeypatch.setattr(ib_api, "ib_connection", lambda: mock_ib)

        async def async_fetch(*args, **kwargs):
            return ibdc._bars_to_df(ibdc.create_mock_bars(1))

        monkeypatch.setattr(ibdc, "_fetch_historical_async", async_fetch)
        symbols = ["AAPL", "GOOG", "MSFT"]
        results = asyncio.run(ibdc.fetch_multiple_historical_data(symbols, "20250501 16:00:00"))
        assert set(results.keys()) == set(symbols)
        assert len(results) == len(symbols)

    def test_fetch_multiple_historical_data_partial_failure(self, monkeypatch, mock_ib):
        from srcPy.data import ib_api
        monkeypatch.setattr(ib_api, "ib_connection", lambda: mock_ib)

        async def async_success(*args, **kwargs):
            return pd.DataFrame(
                {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10], 'average': [1.5],
                 'barCount': [1]},
                index=[pd.Timestamp("2025-01-01", tz="UTC")]
            )

        async def async_failure(*args, **kwargs):
            raise NoDataError("MSFT")

        monkeypatch.setattr(ibdc, "_fetch_historical_async",
                            lambda *args, **kwargs: async_success() if args[0] == "AAPL" else async_failure())
        results = asyncio.run(ibdc.fetch_multiple_historical_data(
            symbols=["AAPL", "MSFT"],
            end_date="20250501 16:00:00"
        ))
        assert "AAPL" in results
        assert "MSFT" not in results

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, monkeypatch, mock_ib):
        from srcPy.data import ib_api
        monkeypatch.setattr(ib_api, "ib_connection", lambda: mock_ib)
        active_tasks = []

        async def slow_fetch(*args, **kwargs):
            active_tasks.append(1)
            await asyncio.sleep(0.1)
            active_tasks.pop()
            return pd.DataFrame(
                {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10], 'average': [1.5],
                 'barCount': [1]},
                index=[pd.Timestamp("2025-01-01", tz="UTC")]
            )

        monkeypatch.setattr(ibdc, "_fetch_historical_async", slow_fetch)
        symbols = [f"SYM{i}" for i in range(10)]
        results = await ibdc.fetch_multiple_historical_data(symbols, "20250501 16:00:00")
        assert max(len(active_tasks) for _ in range(len(symbols))) <= 5
        assert len(results) == 10
