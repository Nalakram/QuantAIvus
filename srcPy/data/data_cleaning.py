import yaml
import pandas as pd
import numpy as np

# Load cleaning config section
with open("data/config.yaml", 'r') as f:
    __full_config = yaml.safe_load(f)
_clean_conf = __full_config.get("cleaning", {})


class DataCleaner:
    """Cleans time series data using configurable strategies for missing values,
    outlier handling, and denoising."""

    def __init__(self, config: dict = None):
        """
        Args:
            config (dict): Cleaning config (section 'cleaning' from config.yaml).
        """
        self.config = config or _clean_conf
        self.missing_cfg = self.config.get("missing_values", {})
        self.outlier_cfg = self.config.get("outliers", {})
        self.denoise_cfg = self.config.get("denoising", {})

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies missing value imputation, outlier correction, and denoising.

        Args:
            df (pd.DataFrame): Raw or partially cleaned data.
        Returns:
            pd.DataFrame: Fully cleaned data.
        """
        # 1. Impute missing values
        method = self.missing_cfg.get("method")
        if method == "forward_fill":
            df = df.fillna(method='ffill')
            if self.missing_cfg.get("params", {}).get("backward_fill"):
                df = df.fillna(method='bfill')
        elif method == "interpolation":
            params = self.missing_cfg.get("params", {})
            df = df.interpolate(method=params.get("method", 'linear'), limit_direction='both')
        elif method == "mean":
            df = df.fillna(df.mean(numeric_only=True))

        # 2. Handle outliers
        o_method = self.outlier_cfg.get("method")
        if o_method == "zscore":
            thresh = self.outlier_cfg.get("params", {}).get("threshold", 3.0)
            num_cols = df.select_dtypes(include=np.number).columns
            z = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std(ddof=0)
            df[num_cols] = df[num_cols].mask(z.abs() > thresh)
            # re-impute
            df = df.fillna(method='ffill') if method == 'forward_fill' else df.interpolate()
        elif o_method == "iqr":
            factor = self.outlier_cfg.get("params", {}).get("factor", 1.5)
            num_cols = df.select_dtypes(include=np.number).columns
            Q1 = df[num_cols].quantile(0.25); Q3 = df[num_cols].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR; upper = Q3 + factor * IQR
            df[num_cols] = df[num_cols].mask((df[num_cols] < lower) | (df[num_cols] > upper))
            df = df.fillna(method='ffill') if method == 'forward_fill' else df.interpolate()
        elif o_method == "hampel":
            window = self.outlier_cfg.get("params", {}).get("window", 5)
            sigma = self.outlier_cfg.get("params", {}).get("n_sigma", 3)
            num_cols = df.select_dtypes(include=np.number).columns
            for col in num_cols:
                ser = df[col]
                med = ser.rolling(window, center=True).median()
                mad = ser.rolling(window, center=True).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
                thresh = sigma * 1.4826 * mad
                df[col] = ser.mask((ser - med).abs() > thresh)
            df = df.fillna(method='ffill') if method == 'forward_fill' else df.interpolate()

        # 3. Denoise
        d_method = self.denoise_cfg.get("method")
        if d_method == "ewm":
            span = self.denoise_cfg.get("params", {}).get("span", 5)
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].apply(lambda x: x.ewm(span=span).mean())
        elif d_method == "kalman":
            try:
                from pykalman import KalmanFilter
            except ImportError:
                pass
            else:
                num_cols = df.select_dtypes(include=np.number).columns
                for col in num_cols:
                    arr = df[col].values
                    mask = np.isnan(arr)
                    if mask.any():
                        arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
                    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                                      initial_state_mean=arr[0], initial_state_covariance=1,
                                      observation_covariance=1, transition_covariance=0.01)
                    state_means, _ = kf.smooth(arr)
                    df[col] = state_means.flatten()

        return df