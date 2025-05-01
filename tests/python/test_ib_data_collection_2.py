import asyncio
from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import Mock

import python.data.ib_data_collection as ibdc
from python.utils.exceptions import DataFetchError, IBConnectionError, NoDataError

class TestBarsToDf:
    @pytest.mark.parametrize("mutate, error_msg", [
        (lambda bars: bars.clear(), "Empty DataFrame"),
        (lambda bars: setattr(bars[0], "open", None), "Missing expected columns"),
    ])
    def test_bars_to_df_errors(self, mutate):
        bars = ibdc.create_mock_bars(3)
        mutate(bars)
        with pytest.raises(DataFetchError):
            ibdc._bars_to_df(bars)

    def test_bars_to_df_success(self):
        bars = ibdc.create_mock_bars(3)
        df = ibdc._bars_to_df(bars)
        assert list(df.columns) == ["open", "high", "low", "close", "volume", "barCount", "average"]
        assert len(df) == 3

    def test_bars_to_df_duplicate_index(self):
        bars = ibdc.create_mock_bars(2, start_date="2025-01-01")
        bars.append(bars[0])
        df = ibdc._bars_to_df(bars)
        assert len(df) == 2
        assert df.index.duplicated().sum() == 0

    def test_bars_to_df_nan_handling(self, caplog):
        bars = ibdc.create_mock_bars(2)
        bars[1].open = None
        with caplog.at_level("WARNING"):
            df = ibdc._bars_to_df(bars)
        assert "Missing values detected in DataFrame; filling with forward fill" in caplog.text
        assert not df["open"].isna().any()

class TestFetchHistoricalData:
    def test_cache_hit(self, tmp_path, mock_ib, monkeypatch):
        df_cached = pd.DataFrame(
            {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10]},
            index=[pd.Timestamp("2025-01-01")]
        )
        cache_file = tmp_path / "AAPL.parquet"
        df_cached.to_parquet(cache_file)
        monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
        monkeypatch.setattr(pd, "read_parquet", lambda path: df_cached)
        df = ibdc.fetch_historical_data(
            symbol="AAPL",
            end_date="20250102 16:00:00",
            ib_client=mock_ib,
            use_cache=True
        )
        mock_ib.reqHistoricalData.assert_not_called()
        pd.testing.assert_frame_equal(df, df_cached)

    def test_cache_update(self, tmp_path, mock_ib, monkeypatch):
        df_cached = pd.DataFrame(
            {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10]},
            index=[pd.Timestamp("2025-01-01")]
        )
        cache_file = tmp_path / "AAPL.parquet"
        df_cached.to_parquet(cache_file)
        monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
        monkeypatch.setattr(pd, "read_parquet", lambda path: df_cached)
        new_bars = ibdc.create_mock_bars(2, start_date="2025-01-02")
        mock_ib.reqHistoricalData.return_value = new_bars
        df = ibdc.fetch_historical_data(
            symbol="AAPL",
            end_date="20250103 16:00:00",
            ib_client=mock_ib,
            use_cache=True
        )
        assert len(df) == 3
        assert df.index[-1] == pd.Timestamp("2025-01-03")
        updated_cache = pd.read_parquet(cache_file)
        pd.testing.assert_frame_equal(df, updated_cache)

    def test_cache_empty_file(self, tmp_path, mock_ib, monkeypatch):
        cache_file = tmp_path / "AAPL.parquet"
        cache_file.write_bytes(b"")
        monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
        monkeypatch.setattr(pd, "read_parquet", lambda path: (_ for _ in ()).throw(FileNotFoundError()))
        df = ibdc.fetch_historical_data(
            symbol="AAPL",
            end_date="20250430 16:00:00",
            ib_client=mock_ib,
            use_cache=True
        )
        assert not df.empty
        mock_ib.reqHistoricalData.assert_called_once()

    def test_no_cache_fetch(self, mock_ib):
        df = ibdc.fetch_historical_data(
            symbol="AAPL",
            end_date="20250430 16:00:00",
            ib_client=mock_ib,
            use_cache=False
        )
        mock_ib.reqHistoricalData.assert_called_once()
        assert not df.empty

    def test_fetch_historical_data_no_data_with_cache(self, tmp_path, mock_ib, monkeypatch):
        df_cached = pd.DataFrame(
            {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10]},
            index=[pd.Timestamp("2025-01-01")]
        )
        cache_file = tmp_path / "AAPL.parquet"
        df_cached.to_parquet(cache_file)
        monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
        monkeypatch.setattr(pd, "read_parquet", lambda path: df_cached)
        mock_ib.reqHistoricalData.return_value = []
        df = ibdc.fetch_historical_data(
            symbol="AAPL",
            end_date="20250102 16:00:00",
            ib_client=mock_ib,
            use_cache=True
        )
        pd.testing.assert_frame_equal(df, df_cached)

    def test_invalid_date_raises(self, mock_ib):
        with pytest.raises(ValueError):
            ibdc.fetch_historical_data(
                symbol="AAPL",
                end_date="invalid-date",
                ib_client=mock_ib,
                use_cache=False
            )

    def test_invalid_symbol_raises(self, mock_ib, monkeypatch):
        from python.utils.validators import validate_symbol
        monkeypatch.setattr("python.utils.validators.validate_symbol", lambda x: (_ for _ in ()).throw(ValueError("Invalid symbol")))
        with pytest.raises(ValueError, match="Invalid symbol"):
            ibdc.fetch_historical_data(
                symbol="INVALID",
                end_date="20250430 16:00:00",
                ib_client=mock_ib,
                use_cache=False
            )

    def test_missing_config(self, mock_ib, monkeypatch):
        from python.utils.config import config
        monkeypatch.setattr("python.utils.config.config", {})
        with pytest.raises(KeyError):
            ibdc.fetch_historical_data(
                symbol="AAPL",
                end_date="20250430 16:00:00",
                ib_client=mock_ib,
                use_cache=False
            )

    def test_ib_error_bubbles(self, mock_ib_with_error):
        with pytest.raises(IBConnectionError, match="connection lost"):
            ibdc.fetch_historical_data(
                symbol="AAPL",
                end_date="20250430 16:00:00",
                ib_client=mock_ib_with_error,
                use_cache=False
            )

@pytest.mark.asyncio
class TestAsyncHelpers:
    async def test_fetch_historical_async_no_data(self, mock_ib):
        sem = asyncio.Semaphore(1)
        async def empty_bars():
            return []
        mock_ib.reqHistoricalDataAsync.return_value = empty_bars()
        with pytest.raises(NoDataError):
            await ibdc._fetch_historical_async(
                "AAPL", "20250101 00:00:00", "1 D", "1 day",
                mock_ib, sem, "TRADES", True, 1, False
            )

    async def test_fetch_historical_async_caches_and_returns(self, tmp_path, monkeypatch, mock_ib):
        cache_file = tmp_path / "AAPL.parquet"
        monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
        monkeypatch.setattr(pd, "to_parquet", pd.DataFrame.to_parquet)
        sem = asyncio.Semaphore(1)
        bars = ibdc.create_mock_bars(5, start_date="2025-01-01")
        async def async_bars():
            return bars
        mock_ib.reqHistoricalDataAsync.return_value = async_bars()
        df = await ibdc._fetch_historical_async(
            "AAPL", "20250101 00:00:00", "1 D", "1 day",
            mock_ib, sem, "TRADES", True, 1, True
        )
        assert cache_file.exists()
        pd.testing.assert_frame_equal(df, ibdc._bars_to_df(bars))

    async def test_fetch_historical_async_cache_update(self, tmp_path, monkeypatch, mock_ib):
        df_cached = pd.DataFrame(
            {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10]},
            index=[pd.Timestamp("2025-01-01")]
        )
        cache_file = tmp_path / "AAPL.parquet"
        df_cached.to_parquet(cache_file)
        monkeypatch.setattr(ibdc, "_get_cache_path", lambda symbol: cache_file)
        monkeypatch.setattr(pd, "read_parquet", lambda path: df_cached)
        monkeypatch.setattr(pd, "to_parquet", pd.DataFrame.to_parquet)
        new_bars = ibdc.create_mock_bars(2, start_date="2025-01-02")
        async def async_bars():
            return new_bars
        mock_ib.reqHistoricalDataAsync.return_value = async_bars()
        sem = asyncio.Semaphore(1)
        df = await ibdc._fetch_historical_async(
            "AAPL", "20250103 16:00:00", "1 D", "1 day",
            mock_ib, sem, "TRADES", True, 1, True
        )
        assert len(df) == 3
        updated_cache = pd.read_parquet(cache_file)
        pd.testing.assert_frame_equal(df, updated_cache)

    async def test_fetch_historical_async_timeout(self, mock_ib):
        async def async_timeout():
            raise asyncio.TimeoutError("Request timed out")
        mock_ib.reqHistoricalDataAsync.return_value = async_timeout()
        sem = asyncio.Semaphore(1)
        with pytest.raises(DataFetchError, match="Request timed out"):
            await ibdc._fetch_historical_async(
                "AAPL", "20250101 16:00:00", "1 D", "1 day",
                mock_ib, sem, "TRADES", True, 1, False
            )

class TestFetchMultiple:
    def test_fetch_multiple_historical_data(self, monkeypatch, mock_ib):
        import python.data.ib_api
        monkeypatch.setattr(python.data.ib_api, "ib_connection", lambda: mock_ib)
        async def async_fetch(*args, **kwargs):
            return ibdc._bars_to_df(ibdc.create_mock_bars(1))
        monkeypatch.setattr(ibdc, "_fetch_historical_async", async_fetch)
        symbols = ["AAPL", "GOOG", "MSFT"]
        results = ibdc.fetch_multiple_historical_data(symbols, "20250501 16:00:00")
        assert set(results.keys()) == set(symbols)
        assert len(results) == len(symbols)

    def test_fetch_multiple_historical_data_partial_failure(self, monkeypatch, mock_ib):
        import python.data.ib_api
        monkeypatch.setattr(python.data.ib_api, "ib_connection", lambda: mock_ib)
        async def async_success(*args, **kwargs):
            return pd.DataFrame(
                {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10]},
                index=[pd.Timestamp("2025-01-01")]
            )
        async def async_failure(*args, **kwargs):
            raise NoDataError("MSFT")
        monkeypatch.setattr(ibdc, "_fetch_historical_async", lambda *args, **kwargs: async_success() if args[0] == "AAPL" else async_failure())
        results = ibdc.fetch_multiple_historical_data(
            symbols=["AAPL", "MSFT"],
            end_date="20250501 16:00:00"
        )
        assert "AAPL" in results
        assert "MSFT" not in results

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, monkeypatch, mock_ib):
        import python.data.ib_api
        monkeypatch.setattr(python.data.ib_api, "ib_connection", lambda: mock_ib)
        active_tasks = []
        async def slow_fetch(*args, **kwargs):
            active_tasks.append(1)
            await asyncio.sleep(0.1)
            active_tasks.pop()
            return pd.DataFrame(
                {'open': [1], 'high': [2], 'low': [0], 'close': [1.5], 'volume': [10]},
                index=[pd.Timestamp("2025-01-01")]
            )
        monkeypatch.setattr(ibdc, "_fetch_historical_async", slow_fetch)
        symbols = [f"SYM{i}" for i in range(10)]
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ibdc.fetch_multiple_historical_data(symbols, "20250501 16:00:00")
        )
        assert max(len(active_tasks) for _ in range(len(symbols))) <= 5
        assert len(results) == 10