StockPredictionApp
Overview
StockPredictionApp is a machine learning-based application for predicting stock prices using a hybrid Transformer model, integrating data from Interactive Brokers (IB) API, financial statements, and economic indicators. The project combines Python for data processing and model training, C++ for high-performance inference, and Java for a GUI frontend, with gRPC for communication. It includes comprehensive unit and integration tests to ensure reliability.
Features

Data Pipeline: Fetches historical and real-time stock data, fundamental metrics, and economic indicators using ib_insync and other APIs.
Machine Learning: Implements a Transformer-based model with optional LSTM/TCN layers, trained with TensorFlow and exported to ONNX.
Inference: High-performance C++ backend using ONNX Runtime, optimized for NVIDIA GPUs (CUDA 12.9).
GUI: Java-based Swing interface for displaying predictions and metrics.
Testing: Extensive Python unit tests with pytest, C++ tests with Google Test, and Java tests with JUnit.

Project Structure
The project is organized as follows (Version 1.2, 2025-05-01):
StockPredictionApp/
├── python/                       # Python 3.9+ scripts for data pipeline and ML
│   ├── __init__.py              # Makes python/ a package
│   ├── data/                    # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── ib_api.py            # IB API connection
│   │   ├── ib_data_collection.py# Historical/real-time stock data
│   │   ├── fundamental_data.py  # Financial metrics
│   │   ├── market_data.py       # Economic indicators
│   │   ├── specialized_data.py  # Insider trading/ESG data
│   │   └── data_loader.py       # Centralized data fetching
│   ├── models/                  # ML model training
│   │   ├── transformer_model.py # Transformer architecture
│   │   ├── lstm_model.py        # LSTM layer
│   │   ├── tcn_model.py         # TCN layer
│   │   ├── hybrid_model.py      # Transformer + LSTM/TCN
│   │   ├── train_model.py       # Model training
│   │   └── evaluate_model.py    # Model evaluation
│   ├── predict/                 # Prediction logic
│   │   └── make_prediction.py   # gRPC-exposed predictions
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── config.py           # IB API settings
│   │   ├── logger.py           # Logging configuration
│   │   ├── validators.py       # Input validation
│   │   └── exceptions.py       # Custom exceptions
│   └── requirements.txt         # Python dependencies
├── cpp/                         # C++17 backend for inference
│   ├── include/                 # Header files
│   │   ├── api_server.h
│   │   ├── model.h
│   │   ├── data_loader.h
│   │   └── utils.h
│   ├── src/                     # Source files
│   │   ├── api_server.cpp
│   │   ├── model.cpp
│   │   ├── data_loader.cpp
│   │   ├── main.cpp
│   │   └── utils.cpp
│   └── CMakeLists.txt           # Build configuration
├── java/                        # Java 17 GUI frontend
│   ├── src/com/example/
│   │   ├── gui/StockPredictionGUI.java
│   │   └── grpc/BackendClient.java
│   └── pom.xml                  # Maven configuration
├── data/                        # Data storage
│   ├── raw/                     # Unprocessed data
│   │   ├── historical_prices_ib.csv
│   │   ├── financial_statements.csv
│   │   ├── economic_indicators.csv
│   │   └── specialized_data.csv
│   ├── processed/processed_data.bin
│   └── config.yaml              # App configuration
├── models/saved_model.onnx      # Trained model
├── tests/                       # Test suite
│   ├── python/                  # Python unit tests
│   │   ├── conftest.py
│   │   ├── test_ib_data_collection.py
│   │   ├── test_ib_data_collection_2.py
│   │   ├── test_ib_api.py
│   │   ├── test_fundamental_data.py
│   │   ├── test_market_data.py
│   │   ├── test_transformer_model.py
│   │   ├── test_hybrid_model.py
│   │   └── test_make_prediction.py
│   ├── cpp/                     # C++ unit tests
│   │   ├── test_api_server.cpp
│   │   ├── test_model.cpp
│   │   └── test_data_loader.cpp
│   ├── java/test_StockPredictionGUI.java
│   ├── integration/test_end_to_end.py
│   ├── run_tests.sh             # Bash test script
│   └── run_tests.bat            # Windows test script
├── pytest.ini                   # Pytest configuration
├── VERSION.md                   # Version history
├── README.md                    # Project documentation
└── LICENSE                      # MIT License

For details, see StockPredictionApp Directory Structure.markdown.
Prerequisites

Python: 3.9+ (tested with 3.12)
C++: C++17, CUDA 12.9, CMake 3.20+
Java: 17
Tools: Anaconda Prompt, Git Bash (for Bash scripts), Maven, gRPC
Hardware: NVIDIA GPU (e.g., RTX 5090) for C++ inference (optional)

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd StockPredictionApp


Set Up Python Environment:

Create and activate a virtual environment:python -m venv venv
call venv\Scripts\activate


Install dependencies:pip install -r python\requirements.txt




Configure IB API:

Edit data/config.yaml with your IB API credentials (host, port, client_id).
Alternatively, python/utils/config.py provides default settings for testing.


Build C++ Backend:
cd cpp
mkdir build && cd build
cmake ..
cmake --build .


Set Up Java GUI:
cd java
mvn clean install



Running Tests
The project includes Python, C++, and Java unit tests, with a focus on Python tests for the data pipeline.

Python Tests:

In Anaconda Prompt:cd /d D:\Coding_Projects\StockPredictionApp
call venv\Scripts\activate
call tests\run_tests.bat


In Bash (e.g., Git Bash):bash
cd /mnt/d/Coding_Projects/StockPredictionApp
source venv/Scripts/activate
./tests/run_tests.sh


Tests include test_ib_data_collection_2.py and test_ib_api.py, verifying IB API data fetching and connection handling.


C++ and Java Tests:

C++: Run Google Test suite in cpp/build.
Java: Run JUnit tests with mvn test in java/.



Troubleshooting

ModuleNotFoundError: No module named 'python':
Ensure pytest.ini exists with python_paths = python.
Verify __init__.py files in python/, python/data/, and python/utils/.


Dependency Issues:
Check installed packages:pip list | findstr "ib-insync pandas pytest numpy nest-asyncio eventkit"


Reinstall:pip install -r python\requirements.txt




Test Failures:
Run with verbose output:pytest -v tests\python\test_ib_data_collection_2.py tests\python\test_ib_api.py


Share errors for debugging.



Version History
See VERSION.md for changes:

Version 1.2 (2025-05-01): Added LICENSE file (MIT License).
Version 1.1 (2025-04-30): Added config.py, logger.py, pytest.ini, run_tests.bat; updated tests; resolved module errors.
Version 1.0: Initial structure.

Contributing

Submit issues or pull requests to the repository.
Ensure tests pass before submitting changes:call tests\run_tests.bat



License
This project is not open source. All rights reserved.

The software, including source code, trained models, datasets, and modified libraries, is proprietary and confidential.
No part of this project may be used, copied, modified, distributed, or exploited without explicit written permission from the author.

See the LICENSE file for full terms.
