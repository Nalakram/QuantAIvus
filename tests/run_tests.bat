@echo off
set PYTHONPATH=%~dp0..\..
pytest -v tests\python\test_ib_data_collection.py tests\python\test_ib_api.py
REM tests\python\test_fundamental_data.py
REM tests\python\test_market_data.py
REM tests\python\test_transformer_model.py
REM tests\python\test_hybrid_model.py
REM tests\python\test_make_prediction.py
