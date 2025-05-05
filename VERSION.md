# Version History

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
