import yaml
from abc import ABC, abstractmethod
import pandas as pd

# Import cleaning if desired for on-the-fly cleaning
from srcPy.data.data_cleaning import DataCleaner

# Load full configuration
with open("data/config.yaml", 'r') as f:
    __full_config = yaml.safe_load(f)

# Extract data source and cleaning configs
_data_conf = __full_config.get("data_source", {})
_clean_conf = __full_config.get("cleaning", {})


def get_config():
    """
    Returns the entire configuration dictionary loaded from config.yaml.
    """
    return __full_config


class DataLoader(ABC):
    """Base abstract class for data loaders from various sources. """

    def __init__(self, config: dict = None):
        """
        Initialize the DataLoader with a configuration dictionary.
        Args:
            config (dict): Section of the config.yaml for data_source.
        """
        self.config = config or {}

    @abstractmethod
    def load_data(self):
        """Load data according to the source type. Must be implemented by subclasses."""
        pass

    def get_model_data(self, clean: bool = True) -> pd.DataFrame:
        """
        Load raw data and optionally clean it, returning a pandas DataFrame ready for preprocessing.
        Args:
            clean (bool): Whether to apply configured cleaning steps.
        Returns:
            pd.DataFrame: Raw (or cleaned) data.
        """
        data = self.load_data()
        # If chunked or lazy structure, consolidate into DataFrame
        if hasattr(data, "__iter__") and not isinstance(data, pd.DataFrame):
            try:
                data = pd.concat(list(data), ignore_index=True)
            except Exception:
                data = data.compute() if hasattr(data, "compute") else data
        if clean:
            cleaner = DataCleaner(_clean_conf)
            data = cleaner.clean(data)
        return data


class CSVLoader(DataLoader):
    """Loads data from CSV files, supporting chunked or Dask-based loading."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.path = self.config.get("path")
        self.chunksize = self.config.get("chunksize")
        self.use_dask = self.config.get("dask", False)

    def load_data(self):
        """
        Read CSV from disk. Returns iterator if chunksize is set, else full DataFrame.
        """
        if self.use_dask:
            import dask.dataframe as dd
            return dd.read_csv(self.path, blocksize=self.chunksize or None)
        if self.chunksize:
            return pd.read_csv(self.path, chunksize=self.chunksize)
        return pd.read_csv(self.path)


class InfluxDBLoader(DataLoader):
    """Loads data from an InfluxDB time-series database."""

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.host = self.config.get("host")
        self.port = self.config.get("port")
        self.token = self.config.get("token")
        self.org = self.config.get("org")
        self.bucket = self.config.get("bucket")
        self.query = self.config.get("query")
        # Placeholder for InfluxDB client initialization
        # from influxdb_client import InfluxDBClient
        # self.client = InfluxDBClient(url=f"http://{self.host}:{self.port}", token=self.token, org=self.org)

    def load_data(self):
        """
        Executes the configured InfluxDB query and returns a pandas DataFrame.
        """
        # Example using influxdb-client:
        # query_api = self.client.query_api()
        # df = query_api.query_data_frame(self.query)
        # return df
        raise NotImplementedError("InfluxDB loading requires client setup.")


# Convenience factory based on config
def build_loader() -> DataLoader:
    """
    Factory method to construct the appropriate DataLoader based on 'data_source.type'.

    Returns:
        DataLoader: Instance of CSVLoader or InfluxDBLoader.
    """
    source_type = _data_conf.get("type", "CSV").lower()
    if source_type == "csv":
        return CSVLoader(_data_conf)
    if source_type == "influxdb":
        return InfluxDBLoader(_data_conf)
    raise ValueError(f"Unsupported data_source type: {source_type}")


# Example top-level function for users

def load_and_clean_data() -> pd.DataFrame:
    """
    Loads and cleans the data in one call using the configuration file.

    Returns:
        pd.DataFrame: Cleaned data ready for preprocessing.
    """
    loader = build_loader()
    return loader.get_model_data(clean=True)