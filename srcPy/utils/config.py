import json
import os
import re
from pathlib import Path
from typing import Dict, Any
from typing import List
from typing import Optional

import yaml
from jsonschema import validate, ValidationError as SchemaValidationError
from pydantic import BaseModel, ValidationError, validator

from srcPy.utils.exceptions import ConfigValidationError
from srcPy.utils.logger import logger

BASE = Path(__file__).parent.parent
CONFIG_PATH = BASE / "data" / "config.yaml"
SCHEMA_PATH = BASE / "data" / "config_schema.json"

# Preprocessing Section
class RSIConfig(BaseModel):
    enabled: bool
    window: int
    fillna_method: str

class MACDConfig(BaseModel):
    enabled: bool
    fast_period: int
    slow_period: int
    signal_period: int
    fillna_method: str

class TechnicalIndicators(BaseModel):
    rsi: Optional[RSIConfig] = None
    macd: Optional[MACDConfig] = None

class ClipExtremes(BaseModel):
    min: float
    max: float

class NormalizationConfig(BaseModel):
    method: str
    rolling_window: int
    clip_extremes: ClipExtremes

class SentimentConfig(BaseModel):
    enabled: bool
    source: str
    sentiment_model: str

class ESGNormalizedConfig(BaseModel):
    enabled: bool
    method: str

class CustomFeatures(BaseModel):
    sentiment: Optional[SentimentConfig] = None
    esg_normalized: Optional[ESGNormalizedConfig] = None

class Preprocessing(BaseModel):
    technical_indicators: TechnicalIndicators
    normalization: NormalizationConfig
    custom_features: CustomFeatures

# Streaming Section
class Streaming(BaseModel):
    batch_size: int
    update_interval_seconds: int
    buffer_size: int
    max_latency_ms: int
    buffer_retention_seconds: int
    event_triggers: Dict[str, str]
    priority_queue: str
    failure_recovery: Dict[str, Any]
    sync_interval_seconds: int

# Error Handling Section
class RetryPolicy(BaseModel):
    max_attempts: int
    initial_backoff_seconds: int
    max_backoff_seconds: int
    retry_strategy: str

class ValidationThresholds(BaseModel):
    max_missing_ratio: float
    max_outlier_ratio: float

class Fallback(BaseModel):
    twitter: str
    esg: str
    data_source: str

class Alerting(BaseModel):
    enabled: bool
    channel: str
    critical_failures: List[str]
    alert_severity: List[str]

class ErrorHandling(BaseModel):
    retry_policy: RetryPolicy
    validation_thresholds: ValidationThresholds
    fallback: Fallback
    alerting: Alerting
    fallback_timeout_seconds: int

# Data Source Section    
class CSVConfig(BaseModel):
    path: str
    chunksize: int
    use_dask: bool
    compression: str
    data_format: str

class InfluxDBConfig(BaseModel):
    host: str
    port: int
    token: str
    org: str
    bucket: str
    query: str

class DataSource(BaseModel):
    type: str
    csv: Optional[CSVConfig] = None
    influxdb: Optional[InfluxDBConfig] = None

# Alternative Data Section
class TwitterConfig(BaseModel):
    base_url: str
    bearer_token: str
    authentication_type: str
    endpoints: Dict[str, str]
    default_params: Dict[str, Any]
    rate_limit: Dict[str, int]
    retry_after_header: str
    timeout_seconds: int
    cache_duration_hours: int
    data_resolution: str

class ESGConfig(BaseModel):
    base_url: str
    api_key: str
    authentication_type: str
    endpoints: Dict[str, str]
    default_params: Dict[str, str]
    timeout_seconds: int
    cache_duration_hours: int
    data_resolution: str

class AlternativeData(BaseModel):
    twitter: Optional[TwitterConfig] = None
    esg: Optional[ESGConfig] = None
    fred: Optional[dict] = None
    bloomberg: Optional[dict] = None
    weather: Optional[dict] = None
    
# Model Section
class ModelArchitecture(BaseModel):
    num_layers: int
    hidden_size: int
    dropout: float

class Model(BaseModel):
    model_type: str
    architecture: ModelArchitecture
    sequence_length: int
    prediction_horizon: int
    feature_list: List[str]
    training_device: str
    model_checkpoint: Dict[str, int]
    feature_importance: Dict[str, str]
    onnx_export: Dict[str, Any]

# Cleaning Section (Existing)
class Cleaning(BaseModel):
    missing_values: Dict[str, Any]
    outliers: Dict[str, Any]
    denoising: Dict[str, Any]

# Logging Section
class FileOutputConfig(BaseModel):
    enabled: bool
    path: str
    rotation: str

class InfluxDBOutputConfig(BaseModel):
    enabled: bool
    host: str
    port: int
    token: str
    org: str
    bucket: str

class OutputsConfig(BaseModel):
    console: bool
    file: FileOutputConfig
    influxdb: InfluxDBOutputConfig

class MetricAggregationConfig(BaseModel):
    aggregation_window_seconds: int

class DashboardConfig(BaseModel):
    grafana_url: str

class Logging(BaseModel):
    level: str
    outputs: OutputsConfig
    metrics_report_interval_seconds: int
    custom_metrics: List[str]
    model_metrics: List[str]
    metric_aggregation: MetricAggregationConfig
    log_sampling_rate: float
    dashboard_config: DashboardConfig

# Security Section
class EncryptionConfig(BaseModel):
    at_rest: bool
    in_transit: bool
    encryption_algorithm: str

class CredentialsConfig(BaseModel):
    twitter_api_key: str
    esg_api_key: str

class DataAnonymizationConfig(BaseModel):
    anonymize_pii: bool

class ComplianceConfig(BaseModel):
    audit_log: bool
    retention_days: int
    audit_frequency_days: int
    data_anonymization: DataAnonymizationConfig

class Security(BaseModel):
    encryption: EncryptionConfig
    key_management: str
    credentials: CredentialsConfig
    compliance: ComplianceConfig

# Backtesting Section
class RiskManagementConfig(BaseModel):
    stop_loss: float
    max_drawdown: float

class DateRangeConfig(BaseModel):
    start: str
    end: str

class PositionSizingConfig(BaseModel):
    method: str

class Backtesting(BaseModel):
    initial_capital: float
    transaction_cost_rate: float
    slippage_rate: float
    strategy_list: List[str]
    risk_management: RiskManagementConfig
    date_range: DateRangeConfig
    performance_metrics: List[str]
    position_sizing: PositionSizingConfig
    benchmark_index: str
    backtest_frequency: str

# Update Config to include all sections
class Config(BaseModel):
    version: str
    schema_uri: str
    data_source: DataSource
    alternative_data: AlternativeData
    cleaning: Cleaning
    preprocessing: Preprocessing
    streaming: Streaming
    error_handling: ErrorHandling
    model: Model
    logging: Logging
    security: Security
    backtesting: Backtesting

def resolve_env_vars(data):
    if isinstance(data, dict):
        return {k: resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Replace ${VAR} with os.environ['VAR']
        pattern = r'\$\{(\w+)\}'
        matches = re.finditer(pattern, data)
        for match in matches:
            var_name = match.group(1)
            var_value = os.environ.get(var_name, "")
            if not var_value:
                logger.warning(f"Environment variable {var_name} not found; using empty string.")
            data = data.replace(match.group(0), var_value)
        return data
    return data

def load_config() -> Config:
    """
    Load and validate the configuration from the YAML file.
    
    Returns:
        Config: A Pydantic model instance containing the validated configuration.
    
    Raises:
        FileNotFoundError: If the config or schema file is missing.
        ValueError: If validation against the JSON schema or Pydantic model fails.
    """
    if not CONFIG_PATH.exists():
        logger.error(
            "Config file not found",
            error_type="file_not_found",
            path=str(CONFIG_PATH),
            severity="critical"
        )
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    if not SCHEMA_PATH.exists():
        logger.error(
            "Schema file not found",
            error_type="file_not_found",
            path=str(SCHEMA_PATH),
            severity="critical"
        )
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")
    
    with open(CONFIG_PATH, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Resolve environment variables
    config_data = resolve_env_vars(config_data)
    
    # Verify schema_uri matches the actual schema path
    if config_data.get("schema_uri") != SCHEMA_PATH.name:
        logger.error(
            "Schema URI mismatch",
            error_type="schema_mismatch",
            expected=SCHEMA_PATH.name,
            found=config_data.get("schema_uri"),
            severity="critical"
        )
        raise ConfigValidationError("schema_uri in config.yaml does not match the schema file")
    
    # Load JSON schema
    with open(SCHEMA_PATH, 'r') as f:
        schema = json.load(f)
        
    # Validate against JSON schema
    try:
        validate(instance=config_data, schema=schema)
    except SchemaValidationError as e:
        logger.error(
            "JSON schema validation failed",
            error_type="schema_validation_error",
            error_message=str(e),
            severity="critical"
        )
        raise ConfigValidationError(f"Config validation failed: {e.message}", validation_errors=[str(e)])
    
    # Parse with Pydantic
    try:
        config = Config(**config_data)
    except ValidationError as e:
        logger.error(
            "Pydantic validation failed",
            error_type="pydantic_validation_error",
            error_message=str(e),
            severity="critical"
        )
        raise ConfigValidationError(f"Invalid config structure: {e}", validation_errors=e.errors())
    
    logger.info("Configuration loaded successfully", config_version=config.version)
    return config

# Load the configuration once at module level
config = load_config()

def get_config() -> Config:
    """
    Retrieve the loaded configuration object.
    
    Returns:
        Config: The validated configuration as a Pydantic model instance.
    """
    return config