import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import ta  # technical analysis library


class Preprocessor:
    def __init__(self, sequence_length=60, horizon=1, normalization="minmax"):
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.normalization = normalization.lower()
        # Prepare scaler (not fitted yet)
        if self.normalization == "minmax":
            self.scaler = MinMaxScaler()
        elif self.normalization in ["zscore", "standard"]:
            self.scaler = StandardScaler()
        else:
            raise ValueError("Unsupported normalization method")

    def _fill_missing(self, df):
        # Forward-fill missing values in each column (assumes time index is complete or reindexed externally)
        return df.ffill()

    def _add_indicators(self, df):
        # Compute RSI (14-period)
        rsi_indicator = ta.momentum.RSIIndicator(df['Close'], window=14, fillna=True)
        df['RSI'] = rsi_indicator.rsi()
        # Compute MACD (12, 26, 9)
        macd_indicator = ta.trend.MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9, fillna=True)
        df['MACD'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
        df['MACD_hist'] = macd_indicator.macd_diff()  # MACD histogram
        # Compute Bollinger Bands (20-day, 2 std)
        bb_indicator = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2, fillna=True)
        df['BB_upper'] = bb_indicator.bollinger_hband()
        df['BB_lower'] = bb_indicator.bollinger_lband()
        # (Optionally, df['BB_mid'] = bb_indicator.bollinger_mavg())
        return df

    def _normalize_features(self, df):
        # Fit and transform the features using the scaler.
        # We exclude non-feature columns like 'Ticker' or 'Date' if present.
        feature_cols = [col for col in df.columns if col not in ['Ticker', 'Date', 'Timestamp']]
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df

    def _create_sequences(self, data_array, target_array):
        """Slice 2D array into overlapping sequences and corresponding targets."""
        X_list, y_list = [], []
        T = len(data_array)
        seq_len = self.sequence_length
        horiz = self.horizon
        for i in range(0, T - seq_len - horiz + 1):
            X_list.append(data_array[i: i + seq_len])
            # Target is horizon-1 steps ahead after the window
            target_index = i + seq_len + horiz - 1
            y_list.append(target_array[target_index])
        # Convert to numpy arrays
        X_seq = np.array(X_list)
        y_seq = np.array(y_list)
        return X_seq, y_seq

    def transform(self, data):
        """
        Main method to preprocess raw stock data into model-ready sequences.
        Accepts either a pandas DataFrame or a file path to CSV.
        Returns X, y numpy arrays (and optionally a dict for multiple tickers).
        """
        # Load data if a file path is provided
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        results = {}
        # If a ticker column exists, group by ticker; otherwise assume single ticker data
        if 'Ticker' in df.columns or 'Symbol' in df.columns:
            ticker_col = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
            for ticker, df_ticker in df.groupby(ticker_col):
                df_ticker = df_ticker.sort_values(by='Date')  # sort by time if not already
                df_ticker = df_ticker.drop(columns=[ticker_col])  # remove ticker column for processing
                df_ticker = df_ticker.set_index('Date')  # set datetime index if available
                # Fill missing time periods
                df_ticker = self._fill_missing(df_ticker)
                # Add technical indicators
                df_ticker = self._add_indicators(df_ticker)
                # Drop initial rows with NaN (if any remain)
                df_ticker = df_ticker.dropna()
                # Normalize features
                df_ticker = self._normalize_features(df_ticker)
                # Prepare sequences
                feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                                'BB_upper', 'BB_lower']
                data_array = df_ticker[feature_cols].values
                target_array = df_ticker['Close'].values  # predict next close (could use other target)
                X_seq, y_seq = self._create_sequences(data_array, target_array)
                results[ticker] = (X_seq, y_seq)
        else:
            # Single ticker data (no explicit ticker column)
            df = df.sort_values(by='Date')
            df = df.set_index('Date', drop=True)
            df = self._fill_missing(df)
            df = self._add_indicators(df).dropna()
            df = self._normalize_features(df)
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                            'BB_upper', 'BB_lower']
            data_array = df[feature_cols].values
            target_array = df['Close'].values
            X_seq, y_seq = self._create_sequences(data_array, target_array)
            results = (X_seq, y_seq)
        return results
