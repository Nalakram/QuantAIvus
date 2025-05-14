import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas_ta as ta
from srcPy.utils.config import get_config
from srcPy.utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)
config = get_config()


class Preprocessor:
    def __init__(self):
        """Initialize preprocessor with configuration from config.yaml."""
        self.sequence_length = config.model.sequence_length
        self.horizon = config.model.prediction_horizon
        self.normalization = config.preprocessing.normalization.method.lower()
        self.technical_indicators = config.preprocessing.technical_indicators
        self.custom_features = config.preprocessing.custom_features

        # Initialize scaler based on normalization method
        if self.normalization == "minmax":
            self.scaler = MinMaxScaler()
        elif self.normalization in ["zscore", "standard"]:
            self.scaler = StandardScaler()
        else:
            self.scaler = None  # No normalization
            logger.info("No normalization applied", normalization_method=self.normalization)

        # Required columns for OHLCV data
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']

    def _validate_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input DataFrame and standardize column names to lowercase."""
        # Check for required columns
        missing_cols = [col for col in self.required_columns if col.lower() not in [c.lower() for c in df.columns]]
        if missing_cols:
            error_details = {"missing_columns": missing_cols}
            logger.error(
                "Missing required columns in input data",
                error_type="validation_error",
                details=error_details,
                severity="critical"
            )
            raise DataValidationError(f"Missing required columns: {missing_cols}", details=error_details)

        # Standardize column names to lowercase
        df = df.rename(columns={col: col.lower() for col in df.columns})
        logger.info("Standardized column names to lowercase", columns=list(df.columns))
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill missing values in each column."""
        before_na = df.isna().sum().sum()
        df = df.ffill()
        after_na = df.isna().sum().sum()
        logger.info(
            "Filled missing values",
            missing_values_filled=before_na - after_na,
            remaining_na=after_na
        )
        return df

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators based on config.yaml settings."""
        # Compute RSI
        if self.technical_indicators.rsi and self.technical_indicators.rsi.enabled:
            df['rsi'] = ta.rsi(df['close'], length=self.technical_indicators.rsi.window)
            if self.technical_indicators.rsi.fillna_method == "ffill":
                df['rsi'] = df['rsi'].ffill()
            elif self.technical_indicators.rsi.fillna_method == "zero":
                df['rsi'] = df['rsi'].fillna(0)
            logger.info("Added RSI indicator", window=self.technical_indicators.rsi.window)

        # Compute MACD
        if self.technical_indicators.macd and self.technical_indicators.macd.enabled:
            macd = ta.macd(
                df['close'],
                fast=self.technical_indicators.macd.fast_period,
                slow=self.technical_indicators.macd.slow_period,
                signal=self.technical_indicators.macd.signal_period
            )
            df['macd'] = macd[
                f'MACD_{
                    self.technical_indicators.macd.fast_period}_{
                    self.technical_indicators.macd.slow_period}_{
                    self.technical_indicators.macd.signal_period}']
            df['macd_signal'] = macd[
                f'MACDs_{
                    self.technical_indicators.macd.fast_period}_{
                    self.technical_indicators.macd.slow_period}_{
                    self.technical_indicators.macd.signal_period}']
            df['macd_hist'] = macd[
                f'MACDh_{
                    self.technical_indicators.macd.fast_period}_{
                    self.technical_indicators.macd.slow_period}_{
                    self.technical_indicators.macd.signal_period}']
            if self.technical_indicators.macd.fillna_method == "ffill":
                df[['macd', 'macd_signal', 'macd_hist']] = df[['macd', 'macd_signal', 'macd_hist']].ffill()
            elif self.technical_indicators.macd.fillna_method == "zero":
                df[['macd', 'macd_signal', 'macd_hist']] = df[['macd', 'macd_signal', 'macd_hist']].fillna(0)
            logger.info("Added MACD indicator", fast_period=self.technical_indicators.macd.fast_period)

        # Compute ATR
        if self.technical_indicators.atr and self.technical_indicators.atr.enabled:
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.technical_indicators.atr.window)
            if self.technical_indicators.atr.fillna_method == "ffill":
                df['atr'] = df['atr'].ffill()
            elif self.technical_indicators.atr.fillna_method == "zero":
                df['atr'] = df['atr'].fillna(0)
            logger.info("Added ATR indicator", window=self.technical_indicators.atr.window)

        # Compute VWAP
        if self.technical_indicators.vwap and self.technical_indicators.vwap.enabled:
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'],
                                 anchor="D" if self.technical_indicators.vwap.reset_period == "daily" else None)
            if self.technical_indicators.vwap.fillna_method == "ffill":
                df['vwap'] = df['vwap'].ffill()
            elif self.technical_indicators.vwap.fillna_method == "zero":
                df['vwap'] = df['vwap'].fillna(0)
            logger.info("Added VWAP indicator", reset_period=self.technical_indicators.vwap.reset_period)

        # Compute Bollinger Bands
        if self.technical_indicators.bollinger_bands and self.technical_indicators.bollinger_bands.enabled:
            bb = ta.bbands(df['close'], length=self.technical_indicators.bollinger_bands.window,
                           std=self.technical_indicators.bollinger_bands.num_std)
            df['bb_upper'] = bb[f'BBU_{self.technical_indicators.bollinger_bands.window}_{self.technical_indicators.bollinger_bands.num_std}']
            df['bb_lower'] = bb[f'BBL_{self.technical_indicators.bollinger_bands.window}_{self.technical_indicators.bollinger_bands.num_std}']
            if self.technical_indicators.bollinger_bands.fillna_method == "ffill":
                df[['bb_upper', 'bb_lower']] = df[['bb_upper', 'bb_lower']].ffill()
            elif self.technical_indicators.bollinger_bands.fillna_method == "zero":
                df[['bb_upper', 'bb_lower']] = df[['bb_upper', 'bb_lower']].fillna(0)
            logger.info("Added Bollinger Bands", window=self.technical_indicators.bollinger_bands.window)

        return df

    def _process_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process sentiment and ESG features based on config.yaml."""
        # Process sentiment
        if self.custom_features.sentiment and self.custom_features.sentiment.enabled:
            if 'sentiment' not in df.columns:
                logger.warning(
                    "No 'sentiment' column found; assigning neutral sentiment",
                    error_type="missing_column",
                    severity="warning"
                )
                df['sentiment'] = 0.0
            else:
                # Ensure sentiment is in [-1, 1]
                df['sentiment'] = df['sentiment'].clip(-1.0, 1.0)
                logger.info("Processed sentiment feature", avg_sentiment=df['sentiment'].mean())

        # Process ESG normalized scores
        if self.custom_features.esg_normalized and self.custom_features.esg_normalized.enabled:
            if 'esg_score' not in df.columns:
                logger.warning(
                    "No 'esg_score' column found; assigning default score",
                    error_type="missing_column",
                    severity="warning"
                )
                df['esg_score'] = 0.0
            else:
                # Normalize ESG scores using configured method
                if self.custom_features.esg_normalized.method == "minmax":
                    esg_scaler = MinMaxScaler()
                    df['esg_score'] = esg_scaler.fit_transform(df[['esg_score']])
                elif self.custom_features.esg_normalized.method == "zscore":
                    esg_scaler = StandardScaler()
                    df['esg_score'] = esg_scaler.fit_transform(df[['esg_score']])
                logger.info(
                    "Processed ESG normalized feature",
                    method=self.custom_features.esg_normalized.method,
                    avg_esg_score=df['esg_score'].mean()
                )

        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using the configured scaler, excluding non-feature columns."""
        if self.scaler is None:
            logger.info("Skipping normalization", normalization_method=self.normalization)
            return df

        feature_cols = [
            col for col in df.columns
            if col not in ['ticker', 'symbol', 'date', 'timestamp']
            and col in (self.required_columns + ['rsi', 'macd', 'macd_signal', 'macd_hist', 'atr', 'vwap', 'bb_upper', 'bb_lower', 'sentiment', 'esg_score'])
        ]
        if not feature_cols:
            logger.warning("No features to normalize", available_columns=list(df.columns))
            return df

        try:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            logger.info("Normalized features", normalized_columns=feature_cols)
        except Exception as e:
            logger.error(
                "Normalization failed",
                error_type="normalization_error",
                error_message=str(e),
                severity="critical"
            )
            raise DataValidationError(f"Normalization failed: {str(e)}")
        return df

    def _create_sequences(self, data_array: np.ndarray, target_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Slice 2D array into overlapping sequences and corresponding targets."""
        X_list, y_list = [], []
        T = len(data_array)
        seq_len = self.sequence_length
        horiz = self.horizon
        if T < seq_len + horiz:
            logger.error(
                "Data length too short for sequence creation",
                data_length=T,
                required_length=seq_len + horiz,
                severity="critical"
            )
            raise DataValidationError(f"Data length {T} is too short for sequence length {seq_len} and horizon {horiz}")

        for i in range(0, T - seq_len - horiz + 1):
            X_list.append(data_array[i: i + seq_len])
            target_index = i + seq_len + horiz - 1
            y_list.append(target_array[target_index])

        X_seq = np.array(X_list)
        y_seq = np.array(y_list)
        logger.info(
            "Created sequences",
            num_sequences=len(X_seq),
            sequence_length=seq_len,
            horizon=horiz
        )
        return X_seq, y_seq

    def transform(self, data):
        """
        Preprocess raw stock data into model-ready sequences.
        Accepts a pandas DataFrame or a file path to CSV.
        Returns X, y numpy arrays (or a dict for multiple tickers).
        """
        # Load data if a file path is provided
        if isinstance(data, str):
            logger.info("Loading data from CSV", file_path=data)
            df = pd.read_csv(data)
        else:
            df = data.copy()

        results = {}
        # Handle multi-ticker data
        ticker_col = 'ticker' if 'ticker' in df.columns else 'symbol' if 'symbol' in df.columns else None
        if ticker_col:
            for ticker, df_ticker in df.groupby(ticker_col):
                logger.info("Processing ticker", ticker=ticker)
                df_ticker = df_ticker.sort_values(by='date')  # Sort by date
                df_ticker = df_ticker.drop(columns=[ticker_col])  # Remove ticker column
                df_ticker = df_ticker.set_index('date')  # Set datetime index
                df_ticker = self._validate_and_standardize(df_ticker)
                df_ticker = self._fill_missing(df_ticker)
                df_ticker = self._add_indicators(df_ticker)
                df_ticker = self._process_custom_features(df_ticker)
                df_ticker = self._normalize_features(df_ticker)
                df_ticker = df_ticker.dropna()

                # Prepare sequences
                feature_cols = [
                    col for col in df_ticker.columns
                    if col in config.model.feature_list + ['sentiment', 'esg_score']
                ]
                if not feature_cols:
                    logger.error(
                        "No valid feature columns for sequences",
                        available_columns=list(df_ticker.columns),
                        severity="critical"
                    )
                    raise DataValidationError("No valid feature columns for sequence creation")

                data_array = df_ticker[feature_cols].values
                target_array = df_ticker['close'].values  # Predict next close
                try:
                    X_seq, y_seq = self._create_sequences(data_array, target_array)
                    results[ticker] = (X_seq, y_seq)
                except DataValidationError as e:
                    logger.error(
                        "Sequence creation failed for ticker",
                        ticker=ticker,
                        error_message=str(e),
                        severity="critical"
                    )
                    raise
        else:
            # Single ticker data
            logger.info("Processing single ticker data")
            df = df.sort_values(by='date')
            df = df.set_index('date', drop=True)
            df = self._validate_and_standardize(df)
            df = self._fill_missing(df)
            df = self._add_indicators(df)
            df = self._process_custom_features(df)
            df = self._normalize_features(df)
            df = df.dropna()

            feature_cols = [
                col for col in df.columns
                if col in config.model.feature_list + ['sentiment', 'esg_score']
            ]
            if not feature_cols:
                logger.error(
                    "No valid feature columns for sequences",
                    available_columns=list(df.columns),
                    severity="critical"
                )
                raise DataValidationError("No valid feature columns for sequence creation")

            data_array = df[feature_cols].values
            target_array = df['close'].values
            try:
                X_seq, y_seq = self._create_sequences(data_array, target_array)
                results = (X_seq, y_seq)
            except DataValidationError as e:
                logger.error(
                    "Sequence creation failed",
                    error_message=str(e),
                    severity="critical"
                )
                raise

        logger.info("Preprocessing completed", result_type="multi_ticker" if ticker_col else "single_ticker")
        return results
