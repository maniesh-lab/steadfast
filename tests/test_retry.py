"""
Tests for retry.py — no real API calls. Just fake functions that fail
on purpose, to prove the retry policy behaves correctly.
"""
from steadfast.exceptions import NonRetryableProviderError, RetryableProviderError
from steadfast.retry import with_retry


async def test_retries_until_success():
    attempts = {"count": 0}

    @with_retry(max_attempts=5, min_wait=0.01, max_wait=0.05)
    async def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RetryableProviderError("temporary failure", provider="fake")
        return "success"

    result = await flaky()

    assert result == "success"
    assert attempts["count"] == 3


async def test_gives_up_after_max_attempts():
    attempts = {"count": 0}

    @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.05)
    async def always_fails():
        attempts["count"] += 1
        raise RetryableProviderError("permanent-looking failure", provider="fake")

    try:
        await always_fails()
        assert False, "should have raised"
    except RetryableProviderError:
        pass

    assert attempts["count"] == 3


async def test_non_retryable_error_raises_immediately():
    attempts = {"count": 0}

    @with_retry(max_attempts=5, min_wait=0.01, max_wait=0.05)
    async def bad_request():
        attempts["count"] += 1
        raise NonRetryableProviderError("bad api key", provider="fake")

    try:
        await bad_request()
        assert False, "should have raised"
    except NonRetryableProviderError:
        pass

    assert attempts["count"] == 1