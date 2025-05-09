import pytest
import json
from srcPy.data.ib_api import ib_connection, IBConnectionError
from srcPy.utils.config import config
from srcPy.utils.logger import logger

@pytest.mark.asyncio
async def test_ib_connection_success(mock_ib, mock_config, monkeypatch):
    """
    Test successful connection to Interactive Brokers API.
    """
    monkeypatch.setattr("srcPy.data.ib_api.config", mock_config)
    with ib_connection() as ib:
        assert ib is mock_ib
        ib.connect.assert_called_with(mock_config['ib_api']['host'], mock_config['ib_api']['port'], mock_config['ib_api']['client_id'])

@pytest.mark.asyncio
async def test_ib_connection_failure(mock_ib_with_error, mock_config, monkeypatch, caplog):
    """
    Test connection failure with IBConnectionError.
    """
    monkeypatch.setattr("srcPy.data.ib_api.config", mock_config)
    with pytest.raises(IBConnectionError, match="Failed to connect to IB"):
        with ib_connection():
            pass
    for record in caplog.records:
        try:
            log_data = json.loads(record.message)
            if log_data.get("event") == "Failed to connect to IB after retries":
                assert "error" in log_data
                break
        except json.JSONDecodeError:
            continue
    else:
        assert False, f"Expected error log not found in caplog: {caplog.records}"

@pytest.mark.asyncio
async def test_ib_connection_retry_success(mock_ib, mock_config, monkeypatch):
    mock_ib.connect.side_effect = [ConnectionError("first fail"), None]
    monkeypatch.setattr("srcPy.data.ib_api.config", mock_config)
    with ib_connection() as ib:
        assert ib is mock_ib
        assert mock_ib.connect.call_count == 2