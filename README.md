# QuantAIvus ![Version](https://img.shields.io/badge/version-1.2-blue) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-Proprietary-red)

**StockPredictionApp** is a machine learning application for predicting stock prices using a hybrid Transformer model. It integrates data from Interactive Brokers API, financial statements, and economic indicators, with Python for data processing, C++ for inference, and Java for a GUI.

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
QuantAIvus predicts stock prices using a Transformer-based model, combining real-time stock data, fundamental metrics, and economic indicators. The project leverages:
- **Python**: Data pipeline and model training with TensorFlow.
- **C++**: High-performance inference with ONNX Runtime and CUDA.
- **Java**: Swing-based GUI for predictions and metrics.

> **Note**: This is a proprietary project. See [License](#license) for details.

## Features
- 📊 **Data Pipeline**: Fetches stock data, financial metrics, and economic indicators via `ib_insync`.
- 🤖 **Machine Learning**: Hybrid Transformer model with LSTM/TCN layers, exported to ONNX.
- ⚡ **Inference**: Optimized C++ backend for NVIDIA GPUs (CUDA 12.9).
- 🖼️ **GUI**: Java Swing interface for visualizing predictions.
- 🧪 **Testing**: Unit and integration tests with `pytest`, Google Test, and JUnit.

## Project Structure
<details>
<summary>Click to expand</summary>
<pre>
      StockPredictionApp/
         ├── python/                # Data pipeline and ML (Python 3.9+)
         │   ├── data/              # Data loading (e.g., ib_api.py)
         │   ├── models/            # Transformer, LSTM, TCN models
         │   ├── predict/           # Prediction logic
         │   ├── utils/             # Config, logging, validators
         │   └── requirements.txt   # Dependencies
         ├── cpp/                   # Inference backend (C++17)
         │   ├── include/           # Headers
         │   ├── src/               # Source files
         │   └── CMakeLists.txt     # Build config
         ├── java/                  # GUI frontend (Java 17)
         │   ├── src/com/example/   # GUI and gRPC client
         │   └── pom.xml            # Maven config
         ├── data/                  # Raw and processed data
         ├── models/                # Saved model (saved_model.onnx)
         ├── tests/                 # Python, C++, Java tests
         ├── README.md              # Documentation
         └── LICENSE                # Proprietary license
</pre>
See [StockPredictionApp Directory Structure.markdown](StockPredictionApp%20Directory%20Structure.markdown) for details.
</details>

## Prerequisites
| Requirement | Version | Notes |
|-------------|---------|-------|
| Python      | 3.9+    | Tested with 3.12 |
| C++         | C++17   | Requires CUDA 12.9, CMake 3.20+ |
| Java        | 17      | Maven for GUI |
| Tools       | Git Bash, Anaconda Prompt, gRPC | For scripts and builds |
| Hardware    | NVIDIA GPU (optional) | E.g., RTX 5090 for inference |

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Nalakram/QuantAIvus.git
   cd StockPredictionApp
   ```
2. **Set Up Python Environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   pip install -r python/requirements.txt
   ```
3. **Configure IB API:**
   ```bash
   Edit data/config.yaml with IB API credentials (host, port, client_id).
   Alternatively, use defaults in python/utils/config.py for testing.
   ```

4. **Build C++ Backend:**
   ```bash
   cd cpp
   mkdir build && cd build
   cmake ..
   cmake --build .
   ```
5. **Set Up Java GUI:**
   ```bash
   cd java
   mvn clean install
   ```
## Running Tests
### Python Tests
   - On Windows (Anaconda Prompt):
      ```cmd
      cd /d D:\Coding_Projects/StockPredictionApp
      call venv\Scripts\activate
      call tests\run_tests.bat
      ```
   - On Git Bash:
      ```bash
         cd /mnt/d/Coding_Projects/StockPredictionApp
         source venv/Scripts/activate
         ./tests/run_tests.sh
      ```
### C++ and Java Tests
  - C++: Run Google Test suite in `cpp/build.`
  - Java: Run JUnit tests with `mvn test` in `java/`.

## Troubleshooting
   - ModuleNotFoundError: No module named 'python':
   - Ensure `pytest.ini` includes `python_paths = python`.
   - Verify `__init__.py` in `python/`, `python/data/`, and `python/utils/`.
### Dependency Issues:
   ```bash
      pip list | findstr "ib-insync pandas pytest numpy nest-asyncio eventkit"
      pip install -r python/requirements.txt
   ```
### Test Failures:
   ```bash
      pytest -v tests/python/test_ib_data_collection_2.py tests/python/test_ib_api.py
   ```
## Version History
- 1.2 (2025-05-01): Added proprietary LICENSE. 
- 1.1 (2025-04-30): Added config.py, logger.py, pytest.ini, run_tests.bat.
- 1.0: Initial structure.
> See [VERSION.md](VERSION.md) for details.

## Contributing
   - Submit issues or pull requests to the repository.
   - Run tests before submitting:
      ```cmd
      call tests\run_tests.bat
      ```
## License
   This project is proprietary and not open source. All rights reserved. No part of this project may be used, copied, modified, or distributed without explicit written permission from the author. See  for details.

