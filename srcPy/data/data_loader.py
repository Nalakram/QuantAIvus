import asyncio
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import aiohttp
import dask.dataframe as dd
import mlflow
import pandas as pd
from influxdb_client import InfluxDBClient
from requests_cache import CachedSession
from requests_ratelimiter import LimiterSession

from srcPy.utils.config import get_config
from srcPy.utils.exceptions import IBConnectionError, DataFetchError, NoDataError

config = get_config()
logger = logging.getLogger(__name__)

def log_data_load_metrics(df: pd.DataFrame, source_name: str):
    mlflow.log_metric(f"{source_name}_rows_loaded", len(df))
    mlflow.log_param(f"{source_name}_schema", list(df.columns))

class BaseLoader(ABC):
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass

    @abstractmethod
    async def stream_data(self):
        pass

    async def fetch(self, mode: str = "batch"):
        if mode == "stream":
            async for chunk in self.stream_data():
                yield chunk
        else:
            yield self.load_data()

LOADER_REGISTRY = {}

def register_loader(name):
    def decorator(cls):
        LOADER_REGISTRY[name] = cls
        return cls
    return decorator

@register_loader("csv")
class CSVLoader(BaseLoader):
    def __init__(self, conf):
        self.path = conf.path
        self.chunksize = conf.chunksize
        self.use_dask = conf.use_dask
    def load_data(self):
        if self.use_dask:
            ddf = dd.read_csv(self.path, blocksize=self.chunksize)
            df = ddf.compute()
        else:
            df = pd.read_csv(self.path)
        log_data_load_metrics(df, "csv")
        return df
    async def stream_data(self):
        for chunk in pd.read_csv(self.path, chunksize=self.chunksize):
            log_data_load_metrics(chunk, "csv_stream")
            yield chunk
            await asyncio.sleep(0)  # Yield control to event loop

@register_loader("influxdb")
class InfluxDBLoader(BaseLoader):
    def __init__(self, conf):
        for k in ("host","port","token","org","bucket","query"): 
            if not getattr(conf, k, None):
                raise ValueError(f"Missing InfluxDB config: {k}")
        self.client = InfluxDBClient(
            url=f"http://{conf.host}:{conf.port}", token=conf.token, org=conf.org
        )
        self.query = conf.query
    def load_data(self):
        df = self.client.query_api().query_data_frame(self.query)
        log_data_load_metrics(df, "influxdb")
        return df
    async def stream_data(self):
        query_api = self.client.query_api()
        while True:
            # Poll for new data every sync_interval_seconds
            df = query_api.query_data_frame(self.query)
            if not df.empty:
                log_data_load_metrics(df, "influxdb_stream")
                yield df
            await asyncio.sleep(config.streaming.sync_interval_seconds)

class APIDataLoader(BaseLoader):
    def __init__(self, base_url, endpoints, api_key, cache_hours=1, rate_limit=60):
        self.base_url = base_url
        self.endpoints = endpoints
        self.api_key = api_key
        self.session = CachedSession('api_cache', backend='sqlite', expire_after=3600*cache_hours)
        self.limiter = LimiterSession(per_minute=rate_limit, session=self.session)
        rp = config.error_handling.retry_policy
        self.max_attempts = rp.max_attempts
        self.initial_backoff = rp.initial_backoff_seconds
        self.max_backoff = rp.max_backoff_seconds
        self.fallback = config.error_handling.fallback

    async def _load_cached_data(self, source_name: str) -> pd.DataFrame:
        """Load cached data based on fallback configuration."""
        fallback_type = getattr(self.fallback, source_name, None)
        if fallback_type != "cached_data":
            logger.warning(
                "No cached_data fallback configured",
                source=source_name,
                fallback_type=fallback_type,
                severity="warning"
            )
            return pd.DataFrame()
        cache_path = Path("data/cache") / f"{source_name}_cache.json"
        try:
            if cache_path.exists():
                df = pd.read_json(cache_path, orient="records")
                logger.info(
                    "Loaded cached data",
                    source=source_name,
                    path=str(cache_path),
                    rows=len(df)
                )
                log_data_load_metrics(df, f"{source_name}_cached")
                return df
            else:
                logger.warning(
                    "Cache file not found",
                    source=source_name,
                    path=str(cache_path),
                    severity="warning"
                )
                return pd.DataFrame()
        except Exception as e:
            logger.error(
                "Failed to load cached data",
                source=source_name,
                path=str(cache_path),
                error_message=str(e),
                severity="error"
            )
            return pd.DataFrame()

    async def _request(self, session: aiohttp.ClientSession, name, params=None):
        url = self.base_url + self.endpoints[name]
        params = params or {}
        for attempt in range(self.max_attempts):
            try:
                async with session.get(
                    url,
                    params={**params, 'apikey': self.api_key},
                    timeout=config.error_handling.fallback_timeout_seconds
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                logger.warning(
                    "API request failed",
                    error_type="api_request_error",
                    attempt=attempt + 1,
                    max_attempts=self.max_attempts,
                    error_message=str(e),
                    severity="warning"
                )
                backoff = min(self.initial_backoff * (2 ** attempt), self.max_backoff)
                await asyncio.sleep(backoff)
        # After retries fail, attempt to load cached data
        source_name = next((k for k, v in LOADER_REGISTRY.items() if isinstance(self, v)), "unknown")
        cached_df = await self._load_cached_data(source_name)
        if not cached_df.empty:
            return cached_df.to_dict(orient="records")
        raise DataFetchError(f"API request failed after {self.max_attempts} attempts for {source_name}")

    async def load_data(self, queries: list[dict]) -> pd.DataFrame:
        async with aiohttp.ClientSession() as session:
            tasks = [self._request(session, q['name'], q['params']) for q in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        records = []
        for result in results:
            if isinstance(result, dict):
                records.extend(result.get('data', []))
        df = pd.json_normalize(records)
        log_data_load_metrics(df, "api")
        return df

    async def stream_data(self):
        raise NotImplementedError

@register_loader("twitter")
class TwitterLoader(APIDataLoader):
    def __init__(self, conf):
        super().__init__(
            base_url=conf.base_url,
            endpoints={**conf.endpoints, 'filtered_stream': conf.endpoints.get('filtered_stream')},
            api_key=conf.bearer_token,
            cache_hours=conf.cache_duration_hours,
            rate_limit=conf.rate_limit.max_calls_per_window
        )
    async def load_data(self, query: str = "market", max_pages: int = 5) -> pd.DataFrame:
        queries = [{'name': 'user_timeline', 'params': {'query': query, 'max_results': 100}}]
        df = await super().load_data(queries)
        if df.empty:
            logger.error(
                "No data returned from Twitter API",
                error_type="no_data_error",
                query=query,
                severity="critical"
            )
            raise NoDataError(query)
        return df

    async def stream_data(self):
        url = self.base_url + self.endpoints['filtered_stream']
        headers = {'Authorization': f'Bearer {self.api_key}'}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(
                url,
                timeout=config.error_handling.fallback_timeout_seconds
            ) as resp:
                async for line in resp.content:
                    yield line

@register_loader("alpaca_stream")
class AlpacaStreamLoader(BaseLoader):
    def __init__(self, conf):
        self.url = conf.endpoint
        self.api_key = conf.api_key
        self.api_secret = conf.api_secret
        rp = config.error_handling.retry_policy
        self.max_attempts = rp.max_attempts
        self.initial_backoff = rp.initial_backoff_seconds
        self.max_backoff = rp.max_backoff_seconds
    def load_data(self):
        raise NotImplementedError("Use stream_data for real-time")
    async def stream_data(self):
        import websockets
        headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.api_secret}
        for attempt in range(self.max_attempts):
            try:
                async with websockets.connect(self.url, extra_headers=headers) as ws:
                    await ws.send('{"action":"subscribe","trades":["*"],"quotes":["*"]}')
                    async for msg in ws:
                        yield msg
            except Exception as e:
                logger.error(
                    "WebSocket connection failed",
                    error_type="connection_error",
                    attempt=attempt + 1,
                    max_attempts=self.max_attempts,
                    error_message=str(e),
                    severity="critical"
                )
                backoff = min(self.initial_backoff * (2 ** attempt), self.max_backoff)
                await asyncio.sleep(backoff)
        raise IBConnectionError(f"WebSocket connection failed after {self.max_attempts} attempts")
        
@register_loader("esg")
class ESGLoader(APIDataLoader):
    def __init__(self, conf):
        super().__init__(
            base_url=conf.base_url,
            endpoints=conf.endpoints,
            api_key=conf.api_key,
            cache_hours=conf.cache_duration_hours,
            rate_limit=conf.rate_limit.max_calls_per_window if hasattr(conf, 'rate_limit') else 60
        )

    async def load_data(self, company_ids=None):
        company_ids = company_ids or ["AAPL", "MSFT"]  # Default companies
        queries = [{'name': 'company_score', 'params': {'id': cid, 'version': self.endpoints.default_params.version}} for cid in company_ids]
        return await super().load_data(queries)

@register_loader("fred")
class FREDLoader(APIDataLoader):
    def __init__(self, conf):
        super().__init__(
            base_url=conf.base_url,
            endpoints=conf.endpoints,
            api_key=conf.api_key,
            cache_hours=conf.cache_duration_hours,
            rate_limit=conf.rate_limit.max_calls_per_window if hasattr(conf, 'rate_limit') else 60
        )

    async def load_data(self, series_ids=None):
        series_ids = series_ids or ["GDP", "UNRATE"]  # Default series
        queries = [{'name': 'series', 'params': {'series_id': sid, 'file_type': 'json'}} for sid in series_ids]
        return await super().load_data(queries)

@register_loader("bloomberg")
class BloombergLoader(APIDataLoader):
    def __init__(self, conf):
        super().__init__(
            base_url=conf.base_url,
            endpoints=conf.endpoints,
            api_key=conf.api_key,
            cache_hours=conf.cache_duration_hours,
            rate_limit=conf.rate_limit.max_calls_per_window if hasattr(conf, 'rate_limit') else 60
        )

    async def load_data(self, topics=None):
        topics = topics or ["markets"]
        queries = [{'name': 'news', 'params': {'topic': topic}} for topic in topics]
        return await super().load_data(queries)

@register_loader("weather")
class WeatherLoader(APIDataLoader):
    def __init__(self, conf):
        super().__init__(
            base_url=conf.base_url,
            endpoints=conf.endpoints,
            api_key=conf.api_key,
            cache_hours=conf.cache_duration_hours,
            rate_limit=conf.rate_limit.max_calls_per_window if hasattr(conf, 'rate_limit') else 60
        )

    async def load_data(self, cities=None):
        cities = cities or ["New York", "London"]
        queries = [{'name': 'forecast', 'params': {'q': city}} for city in cities]
        return await super().load_data(queries)

def build_loader(name: str = None) -> BaseLoader:
    section = config.data_source
    dtype = (name or section.type).lower()
    cls = LOADER_REGISTRY.get(dtype)
    if name in LOADER_REGISTRY:
        section = config.data_source if name in ('csv', 'influxdb') else config.alternative_data
        return LOADER_REGISTRY[name](getattr(section, name, section))
    raise ValueError(f"Unsupported loader type: {name}")
    return cls(section)
    
class CompositeLoader(BaseLoader):
    def __init__(self, loaders):
        self.loaders = loaders  # List of BaseLoader instances

    def load_data(self):
        dfs = []
        for loader in self.loaders:
            df = loader.load_data()
            if df.empty:
                logger.warning(f"Loader {loader.__class__.__name__} returned empty data")
                continue
            dfs.append(df)
        
        if not dfs:
            raise ValueError("No data loaded from any source")
        
        # Align timestamps (assuming datetime index)
        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.join(df, how='outer', rsuffix=f'_{df.attrs.get("source", "unknown")}')
        combined = combined.sort_index().fillna(method='ffill')
        log_data_load_metrics(combined, "composite")
        return combined

    async def stream_data(self):
        """Stream data from multiple loaders, aggregating chunks asynchronously."""
        async def stream_from_loader(loader, source_name):
            async for chunk in loader.stream_data():
                # Convert chunk to DataFrame if raw (e.g., JSON from API)
                if not isinstance(chunk, pd.DataFrame):
                    try:
                        data = pd.json_normalize(chunk if isinstance(chunk, dict) else json.loads(chunk))
                        df = pd.DataFrame(data)
                        if 'timestamp' in df.columns:
                            df = df.set_index('timestamp')
                        df.attrs['source'] = source_name
                        yield df
                    except Exception as e:
                        logger.warning(
                            "Failed to parse streaming chunk",
                            source=source_name,
                            error_message=str(e),
                            severity="warning"
                        )
                        continue
                else:
                    chunk.attrs['source'] = source_name
                    yield chunk

        # Create streaming tasks for each loader
        tasks = []
        for loader in self.loaders:
            source_name = next((k for k, v in LOADER_REGISTRY.items() if isinstance(loader, v)), "unknown")
            tasks.append(stream_from_loader(loader, source_name))

        # Aggregate streams
        buffer = []
        async for df in asyncio.as_completed(tasks):
            buffer.append(df)
            if len(buffer) == len(self.loaders):
                # Combine buffered DataFrames
                combined = buffer[0]
                for df in buffer[1:]:
                    combined = combined.join(df, how='outer', rsuffix=f'_{df.attrs.get("source", "unknown")}')
                combined = combined.sort_index().fillna(method='ffill')
                log_data_load_metrics(combined, "composite_stream")
                yield combined
                buffer = []

async def stream_live(name: str):
    loader = build_loader(name)
    async for raw in loader.stream_data():
        print(raw)
