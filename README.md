# MarketMind ![Version](https://img.shields.io/badge/version-1.5.0-blue) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-Proprietary-red)

**MarketMind** is a machine learning application for predicting stock prices using a hybrid Transformer model. It integrates data from Interactive Brokers API, financial statements, economic indicators, and textual data (e.g., news, earnings calls) for enhanced predictions. The project combines Python for data processing and model training, C++ for high-performance inference, and Java for a GUI frontend.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Version History](#version-history)
- [Contributing](#contributing)
- [License](#license)

## Overview
MarketMind predicts stock prices using a Transformer-based model with optional LSTM/TCN layers, leveraging:
- **Python**: Data pipeline, model training, and NLP processing with TensorFlow, BERTopic, and spaCy.
- **C++**: High-performance inference with ONNX Runtime and CUDA.
- **Java**: Swing-based GUI for displaying predictions and metrics.
- **Explainability**: SHAP for feature importance analysis.

The project uses gRPC for communication between components and includes comprehensive tests for reliability.

## Features
- 📊 **Data Pipeline**: Fetches real-time/historical stock data, financial metrics, and economic indicators via `ib_insync` and `yfinance`.
- 📜 **NLP Topic Modeling**: Analyzes financial texts (e.g., news, earnings calls) using `bertopic` and `spacy[transformers]` for sentiment and topic insights.
- 🤖 **Machine Learning**: Hybrid Transformer model with LSTM/TCN layers, trained with TensorFlow and exported to ONNX.
- ⚡ **Inference**: Optimized C++ backend for NVIDIA GPUs (CUDA 12.9).
- 🖼️ **GUI**: Java Swing interface for visualizing predictions and metrics.
- 🔍 **Explainability**: SHAP-based feature importance for model transparency.
- 🧪 **Testing**: Unit and integration tests with `pytest`, Google Test, and JUnit.

## Project Structure
<details>
<summary>Click to expand</summary>
<pre>
MarketMind/
├── srcPy/                       # Python 3.9+ scripts for data pipeline, ML, and NLP
│   ├── __init__.py              # Makes srcPy/ a package
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
│   │   ├── __init__.py          # Makes predict/ a package
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
│   │   └── test_ib_api.py
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
</pre>
See [MarketMind Directory Structure.markdown](MarketMind Directory Structure.markdown) for details.
</details>

## Prerequisites
| Requirement | Version                         | Notes                           |
|-------------|---------------------------------|---------------------------------|
| Python      | 3.9+                            | Tested with 3.12                |
| C++         | C++17                           | Requires CUDA 12.9, CMake 3.20+ |
| Java        | 17                              | Maven for GUI                   |
| Tools       | Git Bash, Anaconda Prompt, gRPC | For scripts and builds          |
| Hardware    | NVIDIA GPU (optional)           | E.g., RTX 5090 for inference    |
| Dependencies| `bertopic==0.17.0`, `spacy==3.8.5`, `shap==0.47.2` | For NLP and explainability |

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Nalakram/MarketMind.git
   cd MarketMind
   ```
2. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   pip install -r srcPy/requirements.txt
   ```
3. **Install NLP Model**:
   ```bash
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.1/en_core_web_trf-3.7.1-py3-none-any.whl
   ```
4. **Configure IB API**:
   - Edit `data/config.yaml` with IB API credentials (host, port, client_id).
   - Alternatively, use defaults in `srcPy/utils/config.py` for testing.
5. **Build C++ Backend**:
   ```bash
   cd cpp
   mkdir build && cd build
   cmake ..
   cmake --build .
   ```
6. **Set Up Java GUI**:
   ```bash
   cd java
   mvn clean install
   ```

## Running Tests
### Python Tests
- On Windows (Anaconda Prompt or PowerShell):
  ```cmd
  cd /d D:\Coding_Projects\MarketMind
  call venv\Scripts\activate
  call tests\run_tests.bat
  ```
- On Git Bash:
  ```bash
  cd /mnt/d/Coding_Projects/MarketMind
  source venv/Scripts/activate
  ./tests/run_tests.sh
  ```
- Tests include `test_ib_data_collection.py` and `test_ib_api.py`.

### C++ and Java Tests
- C++: Run Google Test suite in `cpp/build`.
- Java: Run JUnit tests with `mvn test` in `java/`.

## Troubleshooting
- **ModuleNotFoundError: No module named 'srcPy'**:
  - Ensure `pytest.ini` includes `python_paths = srcPy`.
  - Verify `__init__.py` in `srcPy/`, `srcPy/data/`, `srcPy/utils/`, and `srcPy/predict/`.
- **Dependency Issues**:
  ```bash
  pip list | findstr "ib-insync pandas pytest numpy nest-asyncio eventkit bertopic spacy shap"
  pip install -r srcPy/requirements.txt
  ```
- **Test Failures**:
  ```bash
  pytest -v tests\python\test_ib_data_collection.py tests\python\test_ib_api.py
  ```

## Version History
- **1.5.0 (2025-05-04)**: Added `bertopic`, `spacy[transformers]`, and `shap` for NLP topic modeling and model explainability.
- **1.4.0 (2025-05-04)**: Renamed `python/` to `srcPy/`; updated test scripts.
- **1.3.0 (2025-05-01)**: Renamed project to `MarketMind`.
- **1.2.0 (2025-05-01)**: Added Proprietary License.
- **1.1.0 (2025-04-30)**: Added `config.py`, `logger.py`, `pytest.ini`, `run_tests.bat`.
- **1.0.0**: Initial structure.

See [VERSION.md](VERSION.md) for details.

## Contributing
- Submit issues or pull requests to the repository.
- Run tests before submitting:
  ```cmd
  call tests\run_tests.bat
  ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.