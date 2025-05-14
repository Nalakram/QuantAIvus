import pytest
import yaml
from jsonschema.exceptions import ConfigValidationError, ValidationError

from srcPy.utils.config import get_config, load_config  # Assuming these exist


def test_load_config_valid(tmp_path):
    config_content = """
version: "1.1"
schema_uri: "./config_schema.json"
data_source:
  type: "CSV"
  csv:
    path: "data/raw/historical_prices_ib.csv"
    chunksize: 100000
    use_dask: true
    compression: "gzip"
    data_format: "ohlcv"
cleaning:
  missing_values:
    method: "forward_fill"
    params:
      backward_fill: true
  outliers:
    method: "zscore"
    params:
      threshold: 3.0
  denoising:
    method: "ewm"
    params:
      span: 10
preprocessing:
  technical_indicators:
    rsi:
      enabled: true
      window: 14
      fillna_method: "ffill"
  normalization:
    method: "zscore"
    rolling_window: 252
    clip_extremes:
      min: -3
      max: 3
  custom_features:
    sentiment:
      enabled: true
      source: "twitter"
      sentiment_model: "vader"
  calendar_features:
    day_of_week: true
    is_holiday: true
  fundamental_features:
    earnings_dates: true
    dividends: true
streaming:
  batch_size: 500
  update_interval_seconds: 60
  buffer_size: 1000
  max_latency_ms: 100
  buffer_retention_seconds: 300
  event_triggers:
    market_open: "09:30"
    market_close: "16:00"
  priority_queue: "high_frequency"
  failure_recovery:
    restart_interval_seconds: 300
  sync_interval_seconds: 10
error_handling:
  retry_policy:
    max_attempts: 5
    initial_backoff_seconds: 1
    max_backoff_seconds: 30
    retry_strategy: "exponential"
  validation_thresholds:
    max_missing_ratio: 0.05
    max_outlier_ratio: 0.01
  fallback:
    twitter: "cached_data"
    esg: "last_success"
    data_source: "error"
  alerting:
    enabled: true
    channel: "slack"
    critical_failures: ["data_load", "model_inference"]
    alert_severity: ["critical", "warning"]
  fallback_timeout_seconds: 60
logging:
  level: "INFO"
  outputs:
    console: true
    file:
      enabled: true
      path: "logs/marketmind.log"
      rotation: "daily"
    influxdb:
      enabled: false
      host: "localhost"
      port: 8086
      token: "${INFLUXDB_TOKEN}"
      org: "my-org"
      bucket: "metrics"
  metrics_report_interval_seconds: 300
  custom_metrics:
    - "data_load_time"
    - "outlier_count"
  model_metrics:
    - "prediction_latency"
    - "throughput"
  metric_aggregation:
    aggregation_window_seconds: 60
  log_sampling_rate: 0.1
  dashboard_config:
    grafana_url: "http://localhost:3000"
security:
  encryption:
    at_rest: true
    in_transit: true
    encryption_algorithm: "AES-256"
  key_management: "aws_kms"
  credentials:
    twitter_api_key: "${TWITTER_BEARER_TOKEN}"
    esg_api_key: "${ESG_API_KEY}"
  compliance:
    audit_log: true
    retention_days: 90
    audit_frequency_days: 30
    data_anonymization:
      anonymize_pii: true
model:
  model_type: "transformer"
  architecture:
    num_layers: 2
    hidden_size: 128
    dropout: 0.1
  sequence_length: 60
  prediction_horizon: 1
  feature_list:
    - "open"
    - "high"
    - "low"
    - "close"
    - "volume"
    - "rsi"
    - "macd"
  training_device: "cuda"
  model_checkpoint:
    checkpoint_frequency: 10
  feature_importance:
    method: "shap"
  onnx_export:
    enabled: true
    opset_version: 12
hyperparameter_tuning:
  tuner: "bayesian"
  max_trials: 50
  early_stopping:
    patience: 5
  resource_limits:
    max_cpu_per_trial: 2
    max_gpu_per_trial: 1
  search_space:
    learning_rate:
      min: 1e-5
      max: 1e-2
      scale: "log"
    batch_size:
      values: [32, 64, 128]
    sequence_length:
      values: [30, 60, 90]
  parallel_trials: 4
  metric_objective: "minimize_loss"
  pruning_strategy: "median"
backtesting:
  initial_capital: 100000
  transaction_cost_rate: 0.001
  slippage_rate: 0.0005
  strategy_list:
    - "momentum"
    - "mean_reversion"
  risk_management:
    stop_loss: 0.02
    max_drawdown: 0.1
  date_range:
    start: "2020-01-01"
    end: "2024-12-31"
  performance_metrics:
    - "sharpe_ratio"
    - "max_drawdown"
    - "total_return"
  position_sizing:
    method: "kelly_criterion"
  benchmark_index: "SPY"
  backtest_frequency: "monthly"
anomaly_detection:
  method: "isolation_forest"
  params:
    contamination: 0.01
    refit_interval_days: 30
data_augmentation:
  enabled: true
  noise_level: 0.01
  method: "gan"
  augmentation_ratio: 0.2
distributed_processing:
  framework: "dask"
  num_workers: 4
  memory_per_worker: "4GB"
  cluster_type: "local"
monitoring:
  data_drift:
    enabled: true
    threshold: 0.05
    alert_channel: "slack"
  concept_drift:
    method: "ks_test"
experiment_tracking:
  tool: "mlflow"
  tracking_uri: "http://localhost:5000"
model_registry:
  staging_version: "v1.0"
  promotion_criteria:
    min_accuracy: 0.85
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    monkeypatch.setattr(    srcPy.utils.config, "CONFIG_PATH", tmp_path/"config.yaml")
    monkeypatch.setattr(srcPy.utils.config, "SCHEMA_PATH", tmp_path/"config_schema.json")
    config = load_config(str(config_file))
    assert cfg.version == "1.1"
    assert cfg.data_source.type == "CSV"
    assert config["cleaning"]["missing_values"]["method"] == "forward_fill"
    assert config["preprocessing"]["technical_indicators"]["rsi"]["enabled"] is True
    assert config["monitoring"]["data_drift"]["threshold"] == 0.05

def test_load_config_invalid_yaml(tmp_path):
    bad_yaml = "version: [unclosed list\n"
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(bad_yaml)
    with pytest.raises(ConfigValidationError):
        monkeypatch.setattr(    srcPy.utils.config, "CONFIG_PATH", tmp_path/"config.yaml")
        monkeypatch.setattr(srcPy.utils.config, "SCHEMA_PATH", tmp_path/"config_schema.json")
        load_config(str(config_file))

def test_config_missing_required(tmp_path):
    config_content = """
version: "1.1"
schema_uri: "./config_schema.json"
# Missing data_source
cleaning:
  missing_values:
    method: "forward_fill"
    params:
      backward_fill: true
"""
    config_file = tmp_path / "config_missing.yaml"
    config_file.write_text(config_content)
    with pytest.raises(ConfigValidationError):
        monkeypatch.setattr(    srcPy.utils.config, "CONFIG_PATH", tmp_path/"config.yaml")
        monkeypatch.setattr(srcPy.utils.config, "SCHEMA_PATH", tmp_path/"config_schema.json")
        load_config(str(config_file))

@pytest.mark.parametrize("env_var,value,expected", [
    ("TWITTER_BEARER_TOKEN", "fake_twitter", "fake_twitter"),
    ("INFLUXDB_TOKEN", "fake_influx", "fake_influx"),
    ("ALPACA_KEY", "", ""),  # Empty variable
])
def test_env_var_resolution(tmp_path, monkeypatch, env_var, value, expected):
    monkeypatch.setenv(env_var, value)
    config_content = f"""
version: "1.1"
schema_uri: "./config_schema.json"
data_source:
  type: "CSV"
  csv:
    path: "data/raw/historical_prices_ib.csv"
    chunksize: 100000
    use_dask: true
    compression: "gzip"
    data_format: "ohlcv"
security:
  credentials:
    twitter_api_key: "${{TWITTER_BEARER_TOKEN}}"
logging:
  outputs:
    influxdb:
      enabled: false
      host: "localhost"
      port: 8086
      token: "${{INFLUXDB_TOKEN}}"
      org: "my-org"
      bucket: "metrics"
real_time_market_data:
  alpaca:
    api_key: "${{ALPACA_KEY}}"
"""
    config_file = tmp_path / "config_env.yaml"
    config_file.write_text(config_content)
    monkeypatch.setattr(    srcPy.utils.config, "CONFIG_PATH", tmp_path/"config.yaml")
    monkeypatch.setattr(srcPy.utils.config, "SCHEMA_PATH", tmp_path/"config_schema.json")
    config = load_config(str(config_file))
    if env_var == "TWITTER_BEARER_TOKEN":
        assert config["security"]["credentials"]["twitter_api_key"] == expected
    elif env_var == "INFLUXDB_TOKEN":
        assert config["logging"]["outputs"]["influxdb"]["token"] == expected
    elif env_var == "ALPACA_KEY":
        assert config["real_time_market_data"]["alpaca"]["api_key"] == expected

def test_get_config_singleton(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
version: "1.1"
data_source:
  type: "CSV"
""")
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2

def test_config_schema_validation():
    import json
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("config_schema.json") as f:
        schema = json.load(f)
    from jsonschema import validate
    validate(instance=cfg, schema=schema)  # Should not raise
