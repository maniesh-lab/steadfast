"""
Retry policy for Steadfast.

Retries only RetryableProviderError. NonRetryableProviderError (bad API
key, malformed request) is allowed to raise immediately — retrying it
wastes time and API calls for no benefit.

Provider SDKs' own built-in retries should be disabled (max_retries=0
on the SDK client) so this file is the single place retry behavior is
decided, consistently, across every provider.
"""
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .exceptions import RetryableProviderError
from .logging_utils import get_logger

log = get_logger()


def _log_retry_attempt(retry_state):
    """Called by tenacity right before it sleeps and tries again."""
    exc = retry_state.outcome.exception()
    log.warning(
        "retrying provider call",
        attempt=retry_state.attempt_number,
        error=str(exc),
    )


def with_retry(max_attempts: int = 5, min_wait: float = 1.0, max_wait: float = 30.0):
    """
    Decorator factory. Wraps a function so it retries on
    RetryableProviderError, with exponential backoff + jitter.

    Usage:
        @with_retry(max_attempts=5)
        def call_anthropic(...):
            ...
    """
    return retry(
        retry=retry_if_exception_type(RetryableProviderError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=min_wait, max=max_wait),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )