# Version History

## Version 1.6.0 (2025-05-06)
- **Added Files**:
  - `srcPy/data/alternative_data.py`: Fetches alternative data (social media, supply chain, ESG, insider trading).
  - `srcPy/data/data_cleaning.py`: Implements outlier detection, Kalman filtering, and normalization.
  - `srcPy/models/ensemble/ensemble_model.py`: Combines Transformer with XGBoost and ARIMA for ensemble predictions.
  - `srcPy/models/custom_models.py`: Experiments with proprietary Transformer layers for financial data.
  - `srcPy/strategies/stat_arb.py`: Implements statistical arbitrage trading strategies.
  - `srcPy/strategies/momentum.py`: Implements momentum-based trading strategies.
  - `srcPy/utils/risk_management.py`: Adds Kelly criterion, stop-losses, and drawdown monitoring.
  - `srcPy/utils/portfolio.py`: Implements portfolio optimization and capital limits.
  - `srcPy/trading.py`: Automates trade execution via Interactive Brokers API.
  - `srcPy/backtesting.py`: Supports extensive historical simulations.
  - `srcPy/simulation.py`: Manages paper and live trading simulations.
  - `tests/python/test_alternative_data.py`: Tests alternative data fetching.
  - `tests/python/test_ensemble_model.py`: Tests ensemble model accuracy.
  - `tests/python/test_trading.py`: Tests automated trading logic.
  - `tests/python/test_risk_management.py`: Tests risk management functions.
  - `deployment/influxdb_config.yaml`: Configures InfluxDB for time-series storage.
  - `deployment/docker-compose.yml`: Configures cloud GPU deployment.
  - `docs/onboarding.md`: Provides collaboration and setup guide.
- **Added Directories**:
  - `srcPy/models/ensemble/`: Directory for ensemble model scripts.
  - `srcPy/strategies/`: Directory for trading strategy scripts.
  - `deployment/`: Directory for deployment configurations.
  - `docs/`: Directory for team documentation.
- **Updated Files**:
  - `srcPy/ib_data_collection.py`: Extended to support high-frequency intraday data.
  - `srcPy/data_loader.py`: Integrated with InfluxDB for time-series storage.
  - `srcPy/train_model.py`: Added support for short-term prediction horizons and online learning.
  - `srcPy/evaluate_model.py`: Enhanced with statistical focus using SHAP.
  - `cpp/CMakeLists.txt`: Added optimization flags for HFT inference.
  - `README.md`: Updated project structure and feature descriptions.
  - `MarketMind Directory Structure.md`: Reflected new files and directories.
  - `VERSION.md`: Added entry for version 1.6.0.
- **Notes**:
  - Enhanced project for high-frequency trading with automated execution, risk management, and alternative data.
  - Replaced `processed_data.bin` with InfluxDB for efficient data handling.
  - Added ensemble modeling and proprietary layers to improve prediction accuracy.
  - Version incremented to 1.6.0 (MINOR) per Semantic Versioning for new functionality.

## Version 1.5.4 (2025-05-09)
- **Updated Files**:
  - Updated `tests/python/test_ib_data_collection.py`:
    - Fixed `test_no_data_no_cache` by explicitly importing `NoDataError` from `srcPy.data.ib_data_collection` to resolve type mismatch in `pytest.raises`.
    - Moved `test_fetch_historical_async_no_data` into `TestAsyncHelpers` class and corrected indentation of `test_fetch_historical_async_caches_and_returns`, `test_fetch_historical_async_cache_update`, and `test_fetch_historical_async_timeout` to ensure proper test discovery.
    - Ensured all async tests in `TestAsyncHelpers` have `@pytest.mark.asyncio`.
  - Updated `pytest.ini`:
    - Added `testpaths = tests/python` to restrict test discovery to `MarketMind` tests, preventing errors from unrelated directories (e.g., `tensorflow-onnx`).
    - Set `asyncio_mode = auto` to improve async test discovery reliability.
  - Updated `srcPy/requirements.txt` to include `pyarrow` for parquet file support in `ib_data_collection.py`.
  - Updated `README.md` to reflect version 1.5.4 and document test fixes.
  - Updated `VERSION.md` to include this version entry.
- **Dependencies**:
  - Removed `pytest-structlog==1.1` dependency due to potential interference with `pytest.raises` exception handling.
  - Confirmed compatibility with `pytest==8.3.5`, `pytest-asyncio==0.26.0`, `pytest-mock==3.14.0`, `pytest-cov==6.1.1`, `pandas`, `pyarrow`, `ib_insync==0.9.70`, `structlog`, and existing `srcPy/requirements.txt` dependencies.
- **Notes**:
  - Resolved `ModuleNotFoundError` for `parameterized` and `timeout_decorator` by excluding `tensorflow-onnx` tests via `testpaths`.
  - Fixed test discovery issue for `test_fetch_historical_async_no_data` by placing it in `TestAsyncHelpers` and ensuring `asyncio_mode = auto`.
  - Addressed `NoDataError` import mismatch in `test_no_data_no_cache` and `test_fetch_historical_async_no_data`, ensuring `pytest.raises` correctly matches the exception type.
  - Ensured all 26 tests (23 from `test_ib_data_collection.py`, 3 from `test_ib_api.py`) pass with coverage reporting.
  - Improved test reliability by removing `pytest-structlog` and using standard `structlog` logging.
  - Version incremented to 1.5.4 (PATCH) per Semantic Versioning for bug fixes and test configuration improvements.

## Version 1.5.3 (2025-05-05)
- **Updated Files**:
  - Updated `pytest.ini` to include `asyncio_default_fixture_loop_scope = function` to resolve `PytestDeprecationWarning`.
  - Updated `tests\run_tests.bat` to suppress `DeprecationWarning` from `eventkit`.
  - Updated `tests\python\conftest.py` to use `from . import path_setup` for correct import resolution.
  - Updated `README.md` to include `pytest-asyncio>=0.26.0` dependency, reflect version 1.5.3, and document test fixes.
  - Updated `VERSION.md` to include entries for 1.5.1–1.5.3.
- **Notes**:
  - Fixed `PytestDeprecationWarning` by setting `asyncio_default_fixture_loop_scope` explicitly.
  - Suppressed `eventkit` `DeprecationWarning` in `run_tests.bat` for cleaner test output.
  - Resolved `ModuleNotFoundError: No module named 'path_setup'` by reverting to relative import in `conftest.py`.
  - Confirmed all async tests in `test_ib_data_collection.py` and `test_ib_api.py` pass.
  - Version incremented to 1.5.3 (PATCH) per Semantic Versioning for bug fixes.

## Version 1.5.2 (2025-05-05)
- **Updated Files**:
  - Updated `tests\python\conftest.py` to mock `IB` class from `ib_insync` using `monkeypatch`, fixing test failures.
  - Updated `tests\python\test_ib_data_collection.py` and `test_ib_api.py` to use correct assertions and async tests for `ib_connection()`.
  - Updated `pytest.ini` to use `pythonpath = srcPy` instead of `python_paths` and enable `pytest-asyncio` with `markers = asyncio`.
  - Updated `README.md` and `VERSION.md` to reflect test fixes and version 1.5.2.
- **Notes**:
  - Fixed `AssertionError` and `IBConnectionError` in `test_ib_connection_success` by mocking `IB` class instantiation.
  - Fixed `DID NOT RAISE IBConnectionError` in `test_ib_connection_failure` by correctly setting `connect.side_effect`.
  - Addressed `RuntimeWarning: coroutine 'mock_ib.<locals>.async_bars' was never awaited` by returning mock values directly.
  - Added `pytest-asyncio` to support async tests.
  - Version incremented to 1.5.2 (PATCH) per Semantic Versioning for bug fixes.

## Version 1.5.1 (2025-05-05)
- **Updated Files**:
  - Updated `tests\run_tests.bat` and `tests\run_tests.sh` to use correct test paths (`tests\python\test_ib_data_collection.py`, `tests\python\test_ib_api.py`) instead of `tests\srcPy\`.
  - Updated `README.md` to reflect corrected test paths and increment version to 1.5.1.
  - Updated `VERSION.md` to include version 1.5.1 entry.
- **Notes**:
  - Fixed `ERROR: file or directory not found: tests\srcPy\test_ib_data_collection.py` by updating test paths in `run_tests.bat` and `run_tests.sh`.
  - Version incremented to 1.5.1 (PATCH) per Semantic Versioning for bug fixes.

## Version 1.5.0 (2025-05-04)
- **Updated Files**:
  - Updated `srcPy/requirements.txt` to include:
    - `bertopic` for NLP topic modeling of financial texts (e.g., news, earnings calls).
    - `spacy[transformers]>=3.7` and `en_core_web_trf-3.7.1` for advanced NLP processing.
    - `shap` for model explainability, providing feature importance for the hybrid Transformer model.
  - Updated `README.md` to document new NLP and explainability features, reflect `srcPy/`, and align with version 1.5.0.
  - Updated `VERSION.md` to include this version entry and maintain Semantic Versioning.
- **Notes**:
  - Added NLP topic modeling to enhance stock prediction with sentiment and topic analysis from textual data.
  - Added `shap` for explainable AI, improving transparency of model predictions.
  - Version incremented to 1.5.0 (MINOR) per Semantic Versioning for new backward-compatible functionality.

## Version 1.4.0 (2025-05-04)
- **Updated Files**:
  - Renamed `python/` folder to `srcPy/` to avoid namespace conflicts with Python interpreter and distinguish from Java source.
  - Updated `pytest.ini` to use `pythonpath = srcPy`.
  - Updated imports in `tests\python\conftest.py` and test files to use `srcPy.` instead of `python.`.
  - Updated `run_tests.bat` and `run_tests.sh` to include only existing test files (`test_ib_data_collection.py`, `test_ib_api.py`) and comment out non-existent test files.
  - Updated `README.md`, `VERSION.md`, and `MarketMind Directory Structure.markdown` to document folder rename and project rename to `MarketMind`.
- **Notes**:
  - Renaming `python/` to `srcPy/` resolves `ModuleNotFoundError: No module named 'python'` in `conftest.py`.
  - `srcPy/` aligns with project structure separating Python, Java, and C++ source code.

## Version 1.3.0 (2025-05-01)
- **Updated Files**:
  - Renamed project from `StockPredictionApp` to `MarketMind` for clarity and branding.
  - Renamed `StockPredictionApp Directory Structure.markdown` to `MarketMind Directory Structure.markdown`.
  - Updated `README.md`, `VERSION.md`, and `MarketMind Directory Structure.markdown` to reflect new project name.
- **Notes**:
  - `MarketMind` emphasizes AI-driven market analysis, improving project identity.

## Version 1.2.0 (2025-05-01)
- **Added Files**:
  - `LICENSE`: Added Proprietary License to define project licensing terms.
- **Updated Files**:
  - `README.md`: Replaced license placeholder with reference to `LICENSE` file.
  - `StockPredictionApp Directory Structure.markdown`: Added `LICENSE` to structure and updated version history.
- **Notes**:
  - Added `LICENSE` file to clarify usage and distribution terms under a proprietary License.

## Version 1.1.0 (2025-04-30)
- **Added Files**:
  - `python/__init__.py`, `python/data/__init__.py`, `python/utils/__init__.py`: Added to make directories Python packages, resolving `ModuleNotFoundError: No module named 'python'`.
  - `python/utils/config.py`: Defines IB API settings (host, port, client_id, etc.) for `ib_api.py` and `ib_data_collection.py`.
  - `python/utils/logger.py`: Configures logging with console output, compatible with pytest `caplog`.
  - `python/utils/validators.py`: Validates ticker symbols and date formats for data fetching.
  - `python/utils/exceptions.py`: Defines custom exceptions (`IBConnectionError`, `DataFetchError`, `NoDataError`).
  - `tests/python/conftest.py`: Provides pytest fixtures for mocking IB API and cache.
  - `pytest.ini`: Configures pytest to include `python/` in module path.
  - `tests/run_tests.bat`: Windows batch script for running Python unit tests in Anaconda Prompt.
  - `VERSION.md`: Tracks directory structure changes.
  - `README.md`: Project documentation with setup and testing instructions.
- **Updated Files**:
  - `tests/python/test_ib_data_collection_2.py`: Updated imports to use `python.utils.*` (e.g., `python.utils.exceptions`), fixed tests for correct project structure.
  - `tests/python/test_ib_api.py`: Updated imports to use `python.utils.config` and `python.utils.exceptions`, fixed connection tests.
- **Dependencies**:
  - Installed `ib_insync==0.9.70`, `pytest-mock>=3.10`, `pytest-cov>=4.0`, `numpy>=2.0`, `nest-asyncio`, and `eventkit`, which are compatible with existing `python/requirements.txt`.
- **Notes**:
  - Resolved `ModuleNotFoundError: No module named 'python'` by ensuring `__init__.py` files and `pytest.ini`.
  - Added `run_tests.bat` for compatibility with Anaconda Prompt, complementing `run_tests.sh`.
  - Clarified `utils/` placement under `python/` (not top-level), aligning imports accordingly.
  - Confirmed `requirements.txt` already includes necessary dependencies, avoiding redundant additions.

## Version 1.0.0 (Initial)
- Initial project structure as defined in `StockPredictionApp Directory Structure.markdown`.
- Included `python/`, `cpp/`, `java/`, `data/`, `models/`, and `tests/` directories with core functionality for stock prediction.
- No `pytest.ini`, `__init__.py`, or utility files (`config.py`, `logger.py`, etc.) explicitly defined.