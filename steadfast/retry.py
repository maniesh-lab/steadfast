"""
Retry policy for Steadfast.

Retries only RetryableProviderError. NonRetryableProviderError (bad API
key, malformed request) is allowed to raise immediately: retrying it
wastes time and API calls for no benefit.

Provider SDKs' own built-in retries are disabled (max_retries=0 on the
SDK client) so this file is the single place retry behavior is decided,
consistently, across every provider.

Waiting between tries: exponential backoff with jitter (1s, 2s, 4s...
plus a random bit, capped at max_wait). If the provider says how long
to wait (a Retry-After header on a 429), we wait at least that long,
up to RETRY_AFTER_CAP seconds.
"""

from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .exceptions import RetryableProviderError
from .logging_utils import get_logger

log = get_logger()

# Never sleep longer than this because of a Retry-After header, so a
# provider asking for an hour can't freeze your app.
RETRY_AFTER_CAP = 60.0


def _log_retry_attempt(retry_state):
    """Called by tenacity right before it sleeps and tries again."""
    exc = retry_state.outcome.exception()
    log.warning(
        "retrying provider call",
        provider=getattr(exc, "provider", None),
        attempt=retry_state.attempt_number,
        wait_seconds=round(retry_state.upcoming_sleep, 2),
        error=str(exc),
    )


class _wait_backoff_or_retry_after:
    """Exponential backoff with jitter, but never shorter than the
    provider's Retry-After (capped at RETRY_AFTER_CAP)."""

    def __init__(self, min_wait: float, max_wait: float):
        self.backoff = wait_exponential_jitter(multiplier=min_wait, max=max_wait)

    def __call__(self, retry_state) -> float:
        wait = self.backoff(retry_state)
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            wait = max(wait, min(retry_after, RETRY_AFTER_CAP))
        return wait


def _policy(max_attempts: int, min_wait: float, max_wait: float) -> dict:
    return {
        "retry": retry_if_exception_type(RetryableProviderError),
        "stop": stop_after_attempt(max_attempts),
        "wait": _wait_backoff_or_retry_after(min_wait, max_wait),
        "before_sleep": _log_retry_attempt,
        "reraise": True,
    }


def with_retry(max_attempts: int = 5, min_wait: float = 1.0, max_wait: float = 30.0):
    """
    Decorator factory. Wraps a function so it retries on
    RetryableProviderError, with exponential backoff + jitter.

    max_attempts counts every try, including the first.

    Usage:
        @with_retry(max_attempts=5)
        async def call_anthropic(...):
            ...
    """
    return retry(**_policy(max_attempts, min_wait, max_wait))


def retrying(max_attempts: int = 5, min_wait: float = 1.0, max_wait: float = 30.0) -> AsyncRetrying:
    """The same policy as with_retry, as an object client.py can reuse:

    async for attempt in retrying(...):
        with attempt:
            ...
    """
    return AsyncRetrying(**_policy(max_attempts, min_wait, max_wait))
