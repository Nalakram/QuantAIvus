from typing import Iterator
from contextlib import contextmanager
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ib_insync import IB
from srcPy.utils.config import config
from srcPy.utils.exceptions import IBConnectionError
from srcPy.utils.logger import logger
from pydantic import BaseModel


class IBConfig(BaseModel):
    host: str
    port: int
    client_id: int


ib_cfg = IBConfig(**config['ib_api'])


@contextmanager
def ib_connection() -> Iterator[IB]:
    """
    Context manager for Interactive Brokers API connection.

    Yields:
        IB: Connected IB object for API interactions.

    Raises:
        IBConnectionError: If connection fails after 3 retries.

    Notes:
        Retries up to 3 times with exponential backoff (1-10 seconds) on ConnectionError.
    """
    ib = IB()
    try:
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=1, max=10),
            retry=retry_if_exception_type(ConnectionError),
            before_sleep=lambda retry_state: logger.info("Retrying connection", attempt=retry_state.attempt_number)
        )
        def _connect():
            ib.connect(ib_cfg.host, ib_cfg.port, ib_cfg.client_id)
        _connect()
        yield ib
    except Exception as e:
        logger.error("Failed to connect to IB after retries", error=str(e))
        raise IBConnectionError("Failed to connect to IB") from e
    finally:
        ib.disconnect()
