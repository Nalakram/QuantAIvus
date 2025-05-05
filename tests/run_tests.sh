#!/bin/bash
export PYTHONPATH="\:\D:\Coding_Projects\MarketMind/.."
pytest -v tests/python/test_ib_data_collection.py tests/python/test_ib_api.py
# pytest -v tests/python/test_fundamental_data.py
# pytest -v tests/python/test_market_data.py
# pytest -v tests/python/test_transformer_model.py
# pytest -v tests/python/test_hybrid_model.py
# pytest -v tests/python/test_make_prediction.py
