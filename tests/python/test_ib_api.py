import pytest
from unittest.mock import Mock, patch
from ib_insync import IB
from srcPY.data.ib_api import ib_connection, IBConnectionError
from srcPY.utils.config import config
from srcPY.utils.logger import logger

# Mock configuration
@pytest.fixture
def mock_config():
    return {
        'ib_api': {
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1
        }
    }

def test_ib_connection_success(mock_ib, mock_config, monkeypatch):
    mock_instance = Mock()
    mock_ib.return_value = mock_instance
    monkeypatch.setattr("srcPY.data.ib_api.config", mock_config)
    with ib_connection() as ib:
        assert ib == mock_instance
        mock_instance.connect.assert_called_with('127.0.0.1', 7497, 1)
    mock_instance.disconnect.assert_called_once()

def test_ib_connection_failure(mock_ib, mock_config, monkeypatch, caplog):
    mock_instance = Mock()
    mock_instance.connect.side_effect = ConnectionError("Connection failed")
    mock_ib.return_value = mock_instance
    monkeypatch.setattr("srcPY.data.ib_api.config", mock_config)
    with pytest.raises(IBConnectionError, match="Failed to connect to IB"):
        with ib_connection():
            pass
    assert "Retrying connection: attempt" in caplog.text
    assert "Failed to connect to IB after retries" in caplog.t