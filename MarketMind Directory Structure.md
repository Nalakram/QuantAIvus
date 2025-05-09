# MarketMind Directory Structure

MarketMind/
├── srcPy/                       # Python 3.9+ scripts for data pipeline, ML, and trading
│   ├── data/                    # Data loading and preprocessing
│   │   ├── alternative_data.py  # NEW: Fetches social media, supply chain, ESG, insider data
│   │   ├── data_cleaning.py     # NEW: Outlier detection, Kalman filtering, normalization
│   │   ├── ib_api.py            # Manages IB API connection and authentication using ib_insync
│   │   ├── ib_data_collection.py# MODIFIED: Adds high-frequency intraday data
│   │   ├── fundamental_data.py  # Retrieves financial statements and metrics (e.g., P/E ratio, Debt-to-Equity)
│   │   ├── market_data.py       # Collects economic indicators (e.g., GDP, VIX) from external APIs or CSVs
│   │   ├── specialized_data.py  # Optional: Loads insider trading or ESG data with validation
│   │   └── data_loader.py       # MODIFIED: Integrates with InfluxDB
│   ├── models/                  # ML model training
│   │   ├── ensemble/            # NEW: Ensemble model directory
│   │   │   └── ensemble_model.py# Combines Transformer with XGBoost, ARIMA
│   │   ├── custom_models.py     # NEW: Proprietary Transformer layers
│   │   ├── transformer_model.py # Defines Transformer architecture for stock prediction
│   │   ├── lstm_model.py        # Optional: LSTM refinement layer for local patterns
│   │   ├── tcn_model.py         # Optional: TCN refinement layer for temporal convolutions
│   │   ├── hybrid_model.py      # Combines Transformer with LSTM/TCN for improved accuracy
│   │   ├── train_model.py       # MODIFIED: Short-term horizons, online learning
│   │   └── evaluate_model.py    # MODIFIED: Statistical focus with SHAP
│   ├── predict/                 # Prediction logic
│   │   ├── __init__.py          # Makes predict/ a package
│   │   └── make_prediction.py   # Uses trained model to predict stock prices, exposed via gRPC
│   ├── strategies/              # NEW: Trading strategies
│   │   ├── stat_arb.py          # Statistical arbitrage
│   │   └── momentum.py          # Momentum-based trading
│   ├── utils/                   # Utility functions
│   │   ├── risk_management.py   # NEW: Kelly criterion, stop-losses, drawdown
│   │   ├── portfolio.py         # NEW: Portfolio optimization, capital limits
│   │   ├── config.py            # Defines IB API settings (e.g., host, port, client_id)
│   │   ├── logger.py            # Configures logging for console output, compatible with pytest caplog
│   │   ├── validators.py        # Validates inputs (e.g., ticker symbols, date ranges) with clear errors
│   │   └── exceptions.py        # Defines custom exceptions (e.g., IBConnectionError, DataFetchError)
│   ├── trading.py               # NEW: Automates trade execution via Interactive Brokers API with leverage and risk controls
│   ├── backtesting.py           # NEW: Conducts extensive historical simulations across market conditions
│   ├── simulation.py            # NEW: Manages paper and live trading simulations with performance logging
│   └── requirements.txt         # Lists dependencies (e.g., ib_insync>=0.9.70, tensorflow==2.19.0, pytest-mock)
├── cpp/                         # C++17 backend for inference
│   ├── include/                 # Header files with Doxygen-style comments
│   │   ├── api_server.h         # Defines gRPC server interface for prediction requests
│   │   ├── model.h              # Interface for loading and running ML model inference (supports Transformer, hybrid)
│   │   ├── data_loader.h        # Interface for loading preprocessed data into memory
│   │   └── utils.h              # Utility functions (e.g., logging, error handling)
│   ├── src/                     # Source files with single-responsibility implementations
│   │   ├── api_server.cpp       # Implements gRPC server with thread pool for concurrent requests
│   │   ├── model.cpp            # OPTIMIZED: Sub-millisecond latency with CUDA
│   │   ├── data_loader.cpp      # Loads binary data for inference with validation
│   │   ├── main.cpp             # Entry point: Initializes and runs the gRPC server
│   │   └── utils.cpp            # Implements shared utilities (e.g., string parsing, logging)
│   └── CMakeLists.txt           # MODIFIED: Configures build with gRPC, CUDA, and optimization flags
├── java/                        # Java 17 GUI frontend
│   ├── src/com/example/         # Java source files with Javadoc comments
│   │   ├── gui/
│   │   │   └── StockPredictionGUI.java # Displays predictions and metrics in Swing GUI
│   │   └── grpc/
│   │       └── BackendClient.java      # gRPC client to fetch predictions from C++ backend
│   └── pom.xml                  # Maven config with gRPC and Java dependencies
├── data/                        # Shared data storage
│   ├── raw/                     # Unprocessed data from various sources
│   │   ├── historical_prices_ib.csv  # IB-sourced stock prices (e.g., timestamp, open, close)
│   │   ├── financial_statements.csv   # Fundamental data (e.g., revenue, EPS)
│   │   ├── economic_indicators.csv    # Market data (e.g., interest rates, VIX)
│   │   └── specialized_data.csv       # Optional: Insider trades, ESG scores
│   ├── processed/               # Preprocessed data ready for training/inference
│   │   └── processed_data.bin   # Binary format for efficient loading (replaced by InfluxDB)
│   └── config.yaml              # App config (e.g., IB API endpoint, model type, hyperparameters)
├── models/                      # Trained model storage
│   └── saved_model.onnx         # Exported Transformer or hybrid model in ONNX format
├── tests/                       # Test suite
│   ├── python/                  # Python unit tests with pytest
│   │   ├── test_alternative_data.py  # NEW: Verifies alternative data fetching and processing
│   │   ├── test_ensemble_model.py    # NEW: Tests ensemble model accuracy and weighting
│   │   ├── test_trading.py           # NEW: Ensures automated trading logic and IB API integration
│   │   ├── test_risk_management.py   # NEW: Validates Kelly criterion, stop-losses, and drawdown monitoring
│   │   ├── test_ib_data_collection.py # Verifies IB API data retrieval (e.g., missing data)
│   │   └── test_ib_api.py       # Tests ib_api.py connection handling
│   ├── cpp/                     # C++ unit tests with Google Test
│   │   ├── test_api_server.cpp  # Tests gRPC server request handling
│   │   ├── test_model.cpp       # Verifies model inference accuracy with Transformer/hybrid
│   │   ├── test_data_loader.cpp # Ensures data loading robustness
│   ├── java/                    # Java unit tests with JUnit
│   │   └── test_StockPredictionGUI.java # Tests GUI rendering and gRPC calls
│   ├── integration/             # End-to-end integration tests
│   │   └── test_end_to_end.py   # Validates Python-C++-Java workflow with Transformer model
│   ├── run_tests.sh             # Shell script to execute Python unit tests
│   └── run_tests.bat            # Windows batch script to execute Python unit tests
├── deployment/                  # NEW: Deployment configurations
│   ├── influxdb_config.yaml     # Configures InfluxDB for time-series data storage
│   └── docker-compose.yml       # Configures cloud GPU deployment (e.g., AWS EC2)
├── docs/                        # NEW: Team documentation
│   └── onboarding.md            # Provides collaboration guidelines and setup instructions
├── pytest.ini
├── VERSION.md
├── README.md
└── LICENSE