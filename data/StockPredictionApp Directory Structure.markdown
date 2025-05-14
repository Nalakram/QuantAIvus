StockPredictionApp/
├── python/                       # Python 3.9+ scripts for data pipeline and ML training (TensorFlow 2.19.0, NumPy, pandas)
│   ├── __init__.py              # Added to make python/ a package
│   ├── data/                    # Data loading and preprocessing modules
│   │   ├── __init__.py         # Added to make data/ a package
│   │   ├── ib_api.py            # Manages IB API connection and authentication using ib_insync
│   │   ├── ib_data_collection.py# Fetches historical/real-time stock data from IB API with error handling
│   │   ├── fundamental_data.py  # Retrieves financial statements and metrics (e.g., P/E ratio, Debt-to-Equity)
│   │   ├── market_data.py       # Collects economic indicators (e.g., GDP, VIX) from external APIs or CSVs
│   │   ├── specialized_data.py  # Optional: Loads insider trading or ESG data with validation
│   │   └── data_loader.py       # Centralizes data fetching from all sources, ensuring clean inputs
│   ├── models/                  # Machine learning model training and evaluation
│   │   ├── transformer_model.py # Defines Transformer architecture for stock prediction
│   │   ├── lstm_model.py        # Optional: LSTM refinement layer for local patterns
│   │   ├── tcn_model.py         # Optional: TCN refinement layer for temporal convolutions
│   │   ├── hybrid_model.py      # Combines Transformer with LSTM/TCN for improved accuracy
│   │   ├── train_model.py       # Trains selected model (Transformer, hybrid, etc.) with TensorFlow
│   │   └── evaluate_model.py    # Backtests model performance with metrics (e.g., RMSE, Sharpe Ratio)
│   ├── predict/                 # Prediction generation logic
│   │   └── make_prediction.py   # Uses trained model to predict stock prices, exposed via gRPC
│   ├── utils/                   # Reusable utility functions
│   │   ├── __init__.py         # Added to make utils/ a package
│   │   ├── config.py           # Added: Defines IB API settings (e.g., host, port, client_id)
│   │   ├── logger.py           # Added: Configures logging for console output, compatible with pytest caplog
│   │   ├── validators.py       # Added: Validates inputs (e.g., ticker symbols, date ranges) with clear errors
│   │   └── exceptions.py       # Added: Defines custom exceptions (e.g., IBConnectionError, DataFetchError)
│   └── requirements.txt         # Lists dependencies (e.g., ib_insync>=0.9.70, tensorflow==2.19.0, pytest-mock, pytest-cov)
├── cpp/                         # C++17 backend for high-performance inference (gRPC, CUDA 12.9, CMake 3.20+)
│   ├── include/                 # Header files with Doxygen-style comments
│   │   ├── api_server.h         # Defines gRPC server interface for prediction requests
│   │   ├── model.h              # Interface for loading and running ML model inference (supports Transformer, hybrid)
│   │   ├── data_loader.h        # Interface for loading preprocessed data into memory
│   │   └── utils.h              # Utility functions (e.g., logging, error handling)
│   ├── src/                     # Source files with single-responsibility implementations
│   │   ├── api_server.cpp       # Implements gRPC server with thread pool for concurrent requests
│   │   ├── model.cpp            # Runs inference using ONNX Runtime, optimized for NVIDIA RTX 5090
│   │   ├── data_loader.cpp      # Loads binary data for inference with validation
│   │   ├── main.cpp             # Entry point: Initializes and runs the gRPC server
│   │   └── utils.cpp            # Implements shared utilities (e.g., string parsing, logging)
│   └── CMakeLists.txt           # Configures build with gRPC, CUDA, and cuDNN support
├── java/                        # Java 17 GUI frontend (Maven, gRPC client)
│   ├── src/                     # Java source files with Javadoc comments
│   │   ├── com/
│   │   │   ├── example/
│   │   │   │   ├── gui/
│   │   │   │   │   └── StockPredictionGUI.java # Displays predictions and metrics in Swing GUI
│   │   │   │   └── grpc/
│   │   │   │       └── BackendClient.java     # gRPC client to fetch predictions from C++ backend
│   └── pom.xml                  # Maven config with gRPC and Java dependencies
├── data/                        # Shared data storage
│   ├── raw/                     # Unprocessed data from various sources
│   │   ├── historical_prices_ib.csv  # IB-sourced stock prices (e.g., timestamp, open, close)
│   │   ├── financial_statements.csv   # Fundamental data (e.g., revenue, EPS)
│   │   ├── economic_indicators.csv    # Market data (e.g., interest rates, VIX)
│   │   └── specialized_data.csv       # Optional: Insider trades, ESG scores
│   ├── processed/               # Preprocessed data ready for training/inference
│   │   └── processed_data.bin   # Binary format for efficient loading
│   └── config.yaml              # App config (e.g., IB API endpoint, model type, hyperparameters)
├── models/                      # Trained model storage
│   └── saved_model.onnx         # Exported Transformer or hybrid model in ONNX format
├── tests/                       # Comprehensive test suite
│   ├── python/                  # Python unit tests with pytest
│   │   ├── conftest.py          # Added: Defines pytest fixtures for IB API and cache mocking
│   │   ├── test_ib_data_collection.py   # Verifies IB API data retrieval (e.g., missing data)
│   │   ├── test_ib_data_collection_2.py # Updated: Tests ib_data_collection.py with cache, error handling, and async fetching
│   │   ├── test_ib_api.py       # Updated: Tests ib_api.py connection handling
│   │   ├── test_fundamental_data.py     # Tests financial data parsing and validation
│   │   ├── test_market_data.py          # Ensures economic data loads correctly
│   │   ├── test_transformer_model.py    # Validates Transformer model architecture and training
│   │   ├── test_hybrid_model.py         # Tests hybrid model (Transformer + LSTM/TCN) for accuracy
│   │   └── test_make_prediction.py      # Checks prediction output for edge cases
│   ├── cpp/                     # C++ unit tests with Google Test
│   │   ├── test_api_server.cpp         # Tests gRPC server request handling
│   │   ├── test_model.cpp              # Verifies model inference accuracy with Transformer/hybrid
│   │   └── test_data_loader.cpp        # Ensures data loading robustness
│   ├── java/                    # Java unit tests with JUnit
│   │   └── test_StockPredictionGUI.java # Tests GUI rendering and gRPC calls
│   ├── integration/             # End-to-end integration tests
│   │   └── test_end_to_end.py          # Validates Python-C++-Java workflow with Transformer model
│   ├── run_tests.sh             # Shell script to execute Python unit tests
│   └── run_tests.bat            # Added: Windows batch script to execute Python unit tests in Anaconda Prompt
├── pytest.ini                   # Added: Configures pytest to include python/ in module path
├── VERSION.md                   # Added: Tracks project directory structure versions and changes
├── README.md                    # Added: Project documentation
└── LICENSE                      # Added: MIT License

# Version History
## Version 1.2 (2025-05-01)
- **Added Files**:
  - `LICENSE`: Added MIT License to define project licensing terms.
- **Updated Files**:
  - `README.md`: Replaced license placeholder with reference to `LICENSE` file.
  - `StockPredictionApp Directory Structure.markdown`: Added `LICENSE` to structure and updated version history.
- **Notes**:
  - Added `LICENSE` file to clarify usage and distribution terms under MIT License.

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

## Version 1.0 (Initial)
- Initial project structure as defined in `StockPredictionApp Directory Structure.markdown`.
- Included `python/`, `cpp/`, `java/`, `data/`, `models/`, and `tests/` directories with core functionality for stock prediction.
- No `pytest.ini`, `__init__.py`, or utility files (`config.py`, `logger.py`, etc.) explicitly defined.
