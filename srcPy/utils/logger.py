import logging
import structlog

# 1) Set up the root stdlib logger so that structlog can hook into it.
logging.basicConfig(level=logging.INFO)

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,                # attach level name
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),  # <— use stdlib LoggerFactory
    wrapper_class=structlog.stdlib.BoundLogger,       # <— use the BoundLogger that wraps stdlib
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
