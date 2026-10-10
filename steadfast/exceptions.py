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
    """The response kept failing validation even after every configured
    attempt (including the retries where we fed the problems back to the
    model).

    Attributes:
        attempts: how many attempts were made.
        last_error: the last attempt's problems, joined into one string.
        last_response: the model's last raw reply (or None).
        errors: the last attempt's problems, one string per problem.
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: str | None = None,
        last_response: str | None = None,
        errors: list[str] | None = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error
        self.last_response = last_response
        self.errors = errors if errors is not None else []