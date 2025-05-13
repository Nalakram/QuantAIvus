# MarketMind ![Version](https://img.shields.io/badge/version-1.6.0-blue) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-Proprietary-red) [![Build Status](https://img.shields.io/github/actions/workflow/status/Nalakram/QuantAIvus/ci.yml?branch=main)](https://github.com/Nalakram/QuantAIvus/actions) [![Coverage Status](https://img.shields.io/codecov/c/github/Nalakram/QuantAIvus?label=Coverage)](https://codecov.io/gh/Nalakram/QuantAIvus) ![Java Version](https://img.shields.io/badge/Java-21-blue?style=flat-square&logo=openjdk&logoColor=white) ![OS Support](https://img.shields.io/badge/OS-Windows-informational?style=flat&logo=Windows&logoColor=white&color=blue) [![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://your-docs-site) ![Status](https://img.shields.io/badge/status-active-brightgreen) [![codecov](https://codecov.io/gh/Nalakram/QuantAIvus/graph/badge.svg?token=Q7B5WQGAOV)](https://codecov.io/gh/Nalakram/QuantAIvus)

**MarketMind** is a machine learning application for predicting stock prices, optimized for high-frequency trading (HFT). It uses a hybrid Transformer model with ensemble techniques, integrating diverse data sources such as Interactive Brokers API, financial statements, economic indicators, and alternative data (e.g., social media, ESG, weather). The project combines Python for data processing and model training, C++ for high-performance inference, and Java for a GUI frontend.

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
MarketMind predicts stock prices using a Transformer-based model with optional LSTM/TCN layers, enhanced by ensemble methods (e.g., XGBoost, ARIMA). It leverages:
- **Python**: Data pipeline, model training, and NLP processing with TensorFlow, BERTopic, and spaCy.
- **C++**: High-performance inference with ONNX Runtime and CUDA for sub-millisecond latency.
- **Java**: Swing-based GUI for displaying predictions and metrics.
- **Explainability**: SHAP for feature importance analysis.
- **Trading Automation**: Automated trade execution via Interactive Brokers API with risk management.

The project uses gRPC for communication and includes comprehensive tests for reliability.

## Features
- **Data Pipeline**: Fetches real-time/historical stock data, financial metrics, economic indicators, and alternative data (social media, supply chain, ESG) via `ib_insync`, `yfinance`, and custom APIs.
- **NLP Topic Modeling**: Analyzes financial texts (e.g., news, earnings calls) using `bertopic` and `spacy[transformers]` for sentiment and topic insights.
- **Machine Learning**: Hybrid Transformer model with LSTM/TCN layers, ensemble methods (XGBoost, ARIMA), and proprietary layers, trained with TensorFlow and exported to ONNX.
- **Inference**: Optimized C++ backend for NVIDIA GPUs (CUDA 12.9) with sub-millisecond latency for HFT.
- **GUI**: Java Swing interface for visualizing predictions and metrics.
- **Explainability**: SHAP-based feature importance for model transparency.
- **Testing**: Unit and integration tests with `pytest`, Google Test, and JUnit.
- **Trading Strategies**: Statistical arbitrage and momentum-based strategies for automated HFT.
- **Risk Management**: Kelly criterion, stop-losses, portfolio optimization, and drawdown monitoring.
- **Deployment**: Cloud GPU support (AWS EC2) and InfluxDB for time-series data storage.

## Project Structure
<details>
<summary>Click to expand</summary>
<pre>
MarketMind is organized into modular directories for Python, C++, and Java components, with dedicated folders for data, models, tests, and deployment configurations. Key components include:

- **`srcPy/`**: Python scripts for data pipeline (`data/`), machine learning models (`models/`), trading strategies (`strategies/`), predictions (`predict/`), and utilities (`utils/`), supporting data fetching, model training, and automated trading.
- **`cpp/`**: C++17 backend for high-performance inference using ONNX Runtime and CUDA, optimized for sub-millisecond latency in HFT.
- **`java/`**: Java 21 GUI frontend with Swing for visualizing predictions and metrics, integrated via gRPC.
- **`data/`**: Stores raw and processed data, including stock prices, financial metrics, and alternative data, with InfluxDB integration.
- **`models/`**: Contains trained models in ONNX format.
- **`tests/`**: Comprehensive test suite with Python (`pytest`), C++ (Google Test), and Java (JUnit) tests, including unit and integration tests.
- **`deployment/`**: Configurations for cloud GPU deployment and InfluxDB time-series storage.
- **`docs/`**: Team documentation, including onboarding guidelines.
</pre>
For a detailed directory structure, see [MarketMind Directory Structure.md](MarketMind Directory Structure.md).
</details>

## Prerequisites
| Requirement | Version                         | Notes                           |
|-------------|---------------------------------|---------------------------------|
| Python      | 3.9+                            | Tested with 3.12                |
| C++         | C++17                           | Requires CUDA 12.9, CMake 3.20+ |
| Java        | 21                              | Maven for GUI                   |
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
   pip install pytest-asyncio
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
- Tests include `test_ib_data_collection.py`, `test_ib_api.py`, `test_alternative_data.py`, `test_ensemble_model.py`, `test_trading.py`, and `test_risk_management.py`.

### C++ and Java Tests
- C++: Run Google Test suite in `cpp/build`.
- Java: Run JUnit tests with `mvn test` in `java/`.

## Troubleshooting
- **ModuleNotFoundError: No module named 'srcPy'**:
  - Ensure `pytest.ini` includes `python_paths = srcPy`.
  - Verify `__init__.py` in `srcPy/`, `srcPy/data/`, `srcPy/utils/`, `srcPy/predict/`, and `srcPy/strategies/`.
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
<details>
<summary>Click to expand</summary>
<pre>
- **1.6.0 (2025-05-09)**: Added alternative data, ensemble models, trading strategies, risk management, backtesting, simulation, and deployment configurations. Updated project structure for HFT. Transitioned GUI from Swing to JavaFX, updated Java to 21, expanded Java package structure, and enhanced CI/CD with Codecov for Java coverage.
- **1.5.4 (2025-05-06)**: Fixed test failures in `test_ib_data_collection.py` by correcting `NoDataError` imports and `TestAsyncHelpers` test placement. Resolved `tensorflow-onnx` test discovery errors by setting `testpaths = tests/python` in `pytest.ini`. Ensured all 26 tests pass.
- **1.5.3 (2025-05-05)**: Configured pytest-asyncio with asyncio_default_fixture_loop_scope = function to resolve PytestDeprecationWarning; updated run_tests.bat to suppress eventkit warning; fixed test imports.
- **1.5.2 (2025-05-05)**: Fixed test failures in test_ib_data_collection.py and test_ib_api.py by mocking IB class; enabled pytest-asyncio.
- **1.5.1 (2025-05-05)**: Fixed conftest.py import to use from . import path_setup; corrected test paths in run_tests.bat and run_tests.sh.
- **1.5.0 (2025-05-04)**: Added `bertopic`, `spacy[transformers]`, and `shap` for NLP topic modeling and model explainability.
- **1.4.0 (2025-05-04)**: Renamed `python/` to `srcPy/`; updated test scripts.
- **1.3.0 (2025-05-01)**: Renamed project to `MarketMind`.
- **1.2.0 (2025-05-01)**: Added Proprietary License.
- **1.1.0 (2025-04-30)**: Added `config.py`, `logger.py`, `pytest.ini`, `run_tests.bat`.
- **1.0.0**: Initial structure.
</pre>
See [VERSION.md](VERSION.md) for details.
</details>

## Contributing
- Submit issues or pull requests to the repository.
- Run tests before submitting:
  ```cmd
  call tests\run_tests.bat
  ```

## License
This project is licensed under a proprietary license. Unauthorized copying, modification, distribution, or use of this software is strictly prohibited without prior written permission from the copyright holder.
See the [LICENSE](LICENSE) file for full terms.