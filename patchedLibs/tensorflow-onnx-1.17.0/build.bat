rem Install the package + test deps
python -m pip install -e .[test] --upgrade

rem Run tests with coverage
python -m pytest --cov=tf2onnx

rem Build the wheel
python setup.py bdist_wheel
