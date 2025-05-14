#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

# 1) System prerequisites
apt-get update
apt-get install -y protobuf-compiler libprotoc-dev

# 2) Install your package + runtime deps
pip install --upgrade pip
pip install -e .          # will pull in numpy, onnx, protobuf>=5.26.1

# 3) Install test-only dependencies
pip install pytest pytest-cov graphviz parameterized pyyaml timeout-decorator

# 4) Run tests with coverage
pytest --cov=tf2onnx

# 5) Build the wheel
python setup.py bdist_wheel
