"""
Exception hierarchy for Steadfast.

Every other module raises one of these instead of a bare Exception, so
retry.py can make a clean decision: retry it, or fail fast.
"""


class SteadfastError(Exception):
    """Base class for every exception this package raises.

    Catch this if you just want to know "something in Steadfast failed"
    without caring about the specific reason.
    """


class ProviderError(SteadfastError):
    """Something went wrong while talking to an LLM provider's API."""

    def __init__(self, message: str, provider: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.provider = provider
        self.original_exception = original_exception


class RetryableProviderError(ProviderError):
    """A transient failure — rate limit, 5xx, timeout, connection reset.

    Safe (and expected) to retry. retry.py's policy should catch this.
    """


class NonRetryableProviderError(ProviderError):
    """A permanent failure — bad API key, malformed request, 4xx that isn't
    a rate limit. Retrying this will never succeed, so retry.py should
    let it raise immediately instead of burning attempts on it.
    """


class ValidationRetryExceeded(SteadfastError):
    """The response kept failing Pydantic validation even after every
    configured retry attempt (including the retries where we fed the
    validation error back to the model).
    """

    def __init__(self, message: str, attempts: int, last_error: str | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class RateLimitBackoffExceeded(SteadfastError):
    """The rate limiter was configured with a max wait time, and a call
    would have needed to wait longer than that to stay within limits.
    """