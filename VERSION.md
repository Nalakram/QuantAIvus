# Version History 
 
## Version 1.1 (2025-04-30) 
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
- **Updated Files**: 
  - `tests/python/test_ib_data_collection_2.py`: Updated imports to use `python.utils.*` (e.g., `python.utils.exceptions`), fixed tests for correct project structure. 
  - `tests/python/test_ib_api.py`: Updated imports to use `python.utils.config` and `python.utils.exceptions`, fixed connection tests. 
- **Dependencies**: 
  - Installed `ib_insync==0.9.70`, `pytest-mock, `pytest-cov, `numpy, `nest-asyncio`, and `eventkit`, which are compatible with existing `python/requirements.txt`. 
- **Notes**: 
  - Resolved `ModuleNotFoundError: No module named 'python'` by ensuring `__init__.py` files and `pytest.ini`. 
  - Added `run_tests.bat` for compatibility with Anaconda Prompt, complementing `run_tests.sh`. 
  - Clarified `utils/` placement under `python/` (not top-level), aligning imports accordingly. 
  - Confirmed `requirements.txt` already includes necessary dependencies, avoiding redundant additions. 
 
## Version 1.0 (Initial) 
- Initial project structure as defined in `StockPredictionApp Directory Structure.markdown`. 
- Included `python/`, `cpp/`, `java/`, `data/`, `models/`, and `tests/` directories with core functionality for stock prediction. 
- No `pytest.ini`, `__init__.py`, or utility files (`config.py`, `logger.py`, etc.) explicitly defined. 
