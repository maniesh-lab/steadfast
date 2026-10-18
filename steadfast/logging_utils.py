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

    Which ID is used, in order:
    1. the one you pass in
    2. the one already bound by an outer block (e.g. your web request),
       so one request keeps one ID across many LLM calls
    3. a new random one

    When the block ends, whatever was bound before is put back, so an
    outer request's ID is never lost.

    Usage:
        with correlation_context() as cid:
            log.info("starting call")   # this line auto-includes correlation_id
    """
    current = structlog.contextvars.get_contextvars().get("correlation_id")
    cid = correlation_id or current or str(uuid.uuid4())
    with structlog.contextvars.bound_contextvars(correlation_id=cid):
        yield cid
