"""
Logging setup for Steadfast.

Every log line automatically includes whatever correlation ID is
currently bound, without you having to pass it into every function call.
"""

import logging
import uuid
from contextlib import contextmanager
import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Call this once, near the start of your program (or a test)."""

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "steadfast") -> structlog.typing.FilteringBoundLogger:
    """
    Use this everywhere instead of calling structlog.get_logger directly,
    so the whole package pulls from one consistent setup.
    """
    return structlog.get_logger(name)


@contextmanager
def correlation_context(correlation_id: str | None = None):
    """
    Bind a correlation ID for the duration of a `with` block.

    Usage:
        with correlation_context() as cid:
            log.info("starting call")   # this line auto-includes correlation_id
    """
    cid = correlation_id or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    try:
        yield cid
    finally:
        structlog.contextvars.unbind_contextvars("correlation_id")