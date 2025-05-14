import logging
from abc import ABC, abstractmethod
from collections import deque

import mlflow
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from prometheus_client import Gauge
from pyod.models.ecod import ECOD
from sklearn.ensemble import IsolationForest
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from srcPy.utils.config import get_config
from srcPy.utils.exceptions import DataValidationError

config = get_config()
logger = logging.getLogger(__name__)

# Prometheus metrics
streaming_latency = Gauge(
    "streaming_cleaner_latency",
    "Latency of streaming cleaner"
)
buffer_length = Gauge(
    "streaming_buffer_length",
    "Current length of streaming buffer"
)


class CleaningStep(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class MissingImputer(CleaningStep):
    def __init__(self, method, params):
        self.method = method
        self.params = params

    def apply(self, df):
        before = df.isna().sum().sum()
        if self.method == 'forward_fill':
            df = df.fillna(method='ffill')
            if self.params.get('backward_fill', False):
                df = df.fillna(method='bfill')
        elif self.method == 'interpolate':
            df = df.interpolate(method=self.params.get('method', 'linear'), limit_direction='both')
        mlflow.log_metric('missing_imputed', before - df.isna().sum().sum())
        return df


class OutlierHandler(CleaningStep):
    def __init__(self, method, params):
        self.method = method
        self.params = params

    def apply(self, df):
        num = df.select_dtypes(include=[np.number])
        mask = pd.DataFrame(False, index=df.index, columns=num.columns)
        if self.method == 'zscore':
            z = (num - num.mean()) / num.std(ddof=0)
            mask = z.abs() > self.params.get('threshold', 3)
            df[num.columns] = num.mask(mask)
        elif self.method == 'iqr':
            Q1, Q3 = num.quantile(0.25), num.quantile(0.75)
            IQR = Q3 - Q1
            mask = (num < Q1 - self.params['factor'] * IQR) | (num > Q3 + self.params['factor'] * IQR)
            df[num.columns] = num.mask(mask)
        mlflow.log_metric('outliers_removed', mask.sum().sum())
        return df.fillna(method='ffill') if self.method in ('zscore', 'iqr') else df


class Denoiser(CleaningStep):
    def __init__(self, method, params):
        self.method = method
        self.params = params

    def apply(self, df):
        num = df.select_dtypes(include=[np.number])
        if self.method == 'ewm':
            span = self.params.get('span', 5)
            df[num.columns] = num.apply(lambda x: x.ewm(span=span).mean())
        mlflow.log_metric('denoise_applied', 1)
        return df


class IncrementalRSI:
    def __init__(self, window):
        self.window = window
        self.gains = deque(maxlen=window)
        self.losses = deque(maxlen=window)
        self.prev_price = None

    def update(self, price):
        if self.prev_price is None:
            self.prev_price = price
            return np.nan
        delta = price - self.prev_price
        gain = max(delta, 0)
        loss = max(-delta, 0)
        self.gains.append(gain)
        self.losses.append(loss)
        self.prev_price = price
        if len(self.gains) < self.window:
            return np.nan
        avg_gain = sum(self.gains) / self.window
        avg_loss = sum(self.losses) / self.window
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        return 100 - (100 / (1 + rs))


class IncrementalRSIStep(CleaningStep):
    def __init__(self, window):
        self.rsi = IncrementalRSI(window)

    def apply(self, df):
        df['rsi'] = [self.rsi.update(p) for p in df['close']]
        return df


class IncrementalMACD:
    def __init__(self, fast, slow, signal):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.ema_fast = None
        self.ema_slow = None
        self.macd = None
        self.macd_signal = None

    def update(self, price):
        if self.ema_fast is None:
            self.ema_fast = price
            self.ema_slow = price
            self.macd = 0
            self.macd_signal = 0
            return self.macd, self.macd_signal
        self.ema_fast = (price * (2 / (self.fast + 1))) + (self.ema_fast * (1 - (2 / (self.fast + 1))))
        self.ema_slow = (price * (2 / (self.slow + 1))) + (self.ema_slow * (1 - (2 / (self.slow + 1))))
        self.macd = self.ema_fast - self.ema_slow
        self.macd_signal = (self.macd * (2 / (self.signal + 1))) + (self.macd_signal * (1 - (2 / (self.signal + 1))))
        return self.macd, self.macd_signal


class IncrementalMACDStep(CleaningStep):
    def __init__(self, fast, slow, signal):
        self.macd = IncrementalMACD(fast, slow, signal)

    def apply(self, df):
        out = [self.macd.update(p) for p in df['close']]
        df['macd'], df['macd_signal'] = zip(*out)
        return df


class SentimentExtractor(CleaningStep):
    def __init__(self, cfg):
        self.enabled = cfg.enabled
        self.model = SentimentIntensityAnalyzer() if self.enabled else None

    def apply(self, df):
        if not self.enabled:
            return df
        if 'text' not in df.columns:
            logger.warning(
                "No 'text' column found for sentiment analysis; assigning neutral sentiment",
                error_type="missing_column",
                severity="warning"
            )
            df['sentiment'] = 0.0
            return df
        # Validate text data

        def compute_sentiment(text):
            if not isinstance(text, str) or not text.strip():
                logger.debug(
                    "Invalid or empty text for sentiment analysis; assigning neutral sentiment",
                    text_type=type(text),
                    severity="debug"
                )
                return 0.0
            return self.model.polarity_scores(text)['compound']

        df['sentiment'] = df['text'].apply(compute_sentiment)
        mlflow.log_metric('sentiment_rows_processed', len(df))
        logger.info(
            "Sentiment analysis completed",
            rows_processed=len(df),
            avg_sentiment=df['sentiment'].mean(),
            severity="info"
        )
        return df


class CalendarFeatures(CleaningStep):
    def __init__(self, cfg):
        self.day = cfg.day_of_week
        self.holiday = cfg.is_holiday
        self.calendar = USFederalHolidayCalendar()

    def apply(self, df):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        if self.day:
            df['day_of_week'] = df.index.dayofweek
        if self.holiday:
            holidays = self.calendar.holidays(start=df.index.min(), end=df.index.max())
            df['is_holiday'] = df.index.isin(holidays)
        return df


class AnomalyDetector(CleaningStep):
    def __init__(self, cfg):
        self.enabled = cfg.enabled
        self.contamination = cfg.params.contamination
        self.refit_interval = cfg.params.get('refit_interval_days', 30) * 24 * 60 * 60  # Convert days to seconds
        self.method = cfg.method  # "isolation_forest" as per config.yaml
        if self.enabled:
            if self.method == "isolation_forest":
                self.model = IsolationForest(contamination=self.contamination, random_state=42)
            else:
                raise ValueError(f"Unsupported anomaly detection method: {self.method}")
        else:
            self.model = None
        self.counter = 0

    def apply(self, df):
        if not self.enabled:
            return df
        num = df.select_dtypes(include=[np.number])
        if self.counter % self.refit_interval == 0:
            self.model.fit(num)
        mask = self.model.predict(num) == 1  # IsolationForest: 1 for inliers, -1 for outliers
        mlflow.log_metric('anomalies_removed', (~mask).sum())
        self.counter += len(df)
        return df[mask]


class StreamingIsolationForest:
    def __init__(self, contamination, refit_every, window_size=1000):
        self.contamination = contamination
        self.refit_every = refit_every
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.counter = 0

    def fit(self, data):
        if len(data) > 0:
            self.model.fit(data)

    def predict(self, df):
        if len(df) == 0:
            return np.ones(len(df), dtype=bool)
        num = df.select_dtypes(include=[np.number])
        self.buffer.extend(num.to_numpy())
        self.counter += len(df)
        if self.counter % self.refit_every == 0 and len(self.buffer) >= self.window_size:
            self.fit(np.array(self.buffer))
        if len(self.buffer) < self.window_size:
            return np.ones(len(df), dtype=bool)  # Not enough data to detect anomalies
        return self.model.predict(num) == 1


class StreamingAnomalyStep(CleaningStep):
    def __init__(self, contamination, refit_every):
        self.detector = StreamingIsolationForest(contamination, refit_every)

    def apply(self, df):
        mask = self.detector.predict(df.select_dtypes('number'))
        return df[mask]


class CleanerPipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self, df, distributed=False):
        if distributed and len(df) > config.distributed_processing.min_rows_for_distributed:
            import dask.dataframe as dd
            ddf = dd.from_pandas(df, npartitions=config.distributed_processing.num_workers)
            return ddf.map_partitions(lambda x: self._run_partition(x)).compute()
        return self._run_partition(df)

    def _run_partition(self, df):
        for step in self.steps:
            df = step.apply(df)
        return df


class StreamingCleanerPipeline(CleanerPipeline):
    def __init__(self, steps, buffer_size=100, window=252):
        super().__init__(steps)
        self.buffer = deque(maxlen=window or buffer_size)
        self.buffer_size = buffer_size
        self.rsi = IncrementalRSI(window)
        self.macd = IncrementalMACD(config.preprocessing.technical_indicators.macd.fast_period,
                                    config.preprocessing.technical_indicators.macd.slow_period,
                                    config.preprocessing.technical_indicators.macd.signal_period)

    async def process_stream(self, stream_gen):
        async for record in stream_gen:
            self.buffer.append(record)
            buffer_length.set(len(self.buffer))
            if len(self.buffer) >= self.buffer_size:
                df = pd.DataFrame(list(self.buffer))
                cleaned = self.run(df)
                yield cleaned


class ValidationStep(CleaningStep):
    def __init__(self, required_columns=None):
        self.required_columns = required_columns or ['open', 'high', 'low', 'close', 'volume']

    def apply(self, df):
        # Check for required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            error_details = {"missing_columns": missing_cols}
            logger.error(
                "Missing required columns",
                error_type="validation_error",
                details=error_details,
                severity="critical"
            )
            raise DataValidationError(f"Missing required columns: {missing_cols}", details=error_details)

        # Check for duplicate timestamps (assuming index is datetime)
        if df.index.duplicated().any():
            duplicates = df.index[df.index.duplicated()].tolist()
            error_details = {"duplicate_timestamps": duplicates}
            logger.error(
                "Duplicate timestamps found",
                error_type="validation_error",
                details=error_details,
                severity="critical"
            )
            raise DataValidationError(f"Duplicate timestamps found: {duplicates}", details=error_details)

        # Check for non-negative prices in OHLCV columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns and (df[col] < 0).any():
                error_details = {"column": col, "negative_indices": df.index[df[col] < 0].tolist()}
                logger.error(
                    "Negative values found in column",
                    error_type="validation_error",
                    details=error_details,
                    severity="critical"
                )
                raise DataValidationError(f"Negative values found in column {col}", details=error_details)

        logger.info("Data validation passed", validation_step="completed")
        mlflow.log_metric('validation_passed', 1)
        return df


class RSICalculator(CleaningStep):
    def __init__(self, cfg):
        self.enabled = cfg.enabled
        self.window = cfg.window
        self.fillna_method = cfg.fillna_method

    def apply(self, df):
        if not self.enabled:
            return df
        # Manual RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        if self.fillna_method == 'ffill':
            df['rsi'] = df['rsi'].fillna(method='ffill')
        elif self.fillna_method == 'zero':
            df['rsi'] = df['rsi'].fillna(0)
        return df


class DataCleaner:
    def __init__(self, cfg=None, streaming=False):
        self.cfg = cfg or config
        steps = []
        steps.append(ValidationStep())  # Add validation as the first step
        steps.append(MissingImputer(self.cfg.cleaning.missing_values.method, self.cfg.cleaning.missing_values.params))
        steps.append(OutlierHandler(self.cfg.cleaning.outliers.method, self.cfg.cleaning.outliers.params))
        steps.append(Denoiser(self.cfg.cleaning.denoising.method, self.cfg.cleaning.denoising.params))
        ti = self.cfg.preprocessing.technical_indicators
        if streaming:
            # For real-time, use incremental steps
            if ti.rsi.enabled:
                steps.append(IncrementalRSIStep(ti.rsi.window))
            if ti.macd.enabled:
                steps.append(IncrementalMACDStep(ti.macd.fast_period, ti.macd.slow_period, ti.macd.signal_period))
        else:
            # For batch, use original calculators
            if ti.rsi.enabled:
                steps.append(RSICalculator(ti.rsi))
            if ti.macd.enabled:
                steps.append(IncrementalMACDStep(ti.macd.fast_period, ti.macd.slow_period, ti.macd.signal_period))
        cf = self.cfg.preprocessing.custom_features
        if cf.sentiment and cf.sentiment.enabled:
            steps.append(SentimentExtractor(cf.sentiment))
        cal = self.cfg.preprocessing.calendar_features
        steps.append(CalendarFeatures(cal))
        ad = self.cfg.anomaly_detection
        steps.append(AnomalyDetector(ad))
        self.pipeline = CleanerPipeline(steps)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        with mlflow.start_run(run_name='data_cleaning'):
            mlflow.log_param('initial_rows', len(df))
            df_clean = self.pipeline.run(df, distributed=True)
            mlflow.log_param('cleaned_rows', len(df_clean))
        return df_clean

    def clean_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.clean(df)
