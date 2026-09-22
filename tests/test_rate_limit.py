"""
Tests for rate_limit.py — proves calls actually get throttled to the
configured rate, and that using an unconfigured provider fails loudly.
"""
import time

import pytest

from steadfast.rate_limit import RateLimiter


async def test_calls_within_limit_dont_wait():
    limiter = RateLimiter()
    limiter.configure("fake", rate=5, period=1.0)

    start = time.monotonic()
    for _ in range(5):
        async with limiter.acquire("fake"):
            pass
    elapsed = time.monotonic() - start

    # 5 calls, limit is 5 per second — should all go through basically instantly
    assert elapsed < 0.2


async def test_extra_call_over_limit_has_to_wait():
    limiter = RateLimiter()
    limiter.configure("fake", rate=2, period=0.5)

    start = time.monotonic()
    for _ in range(3):  # 3rd call exceeds the 2-per-0.5s limit
        async with limiter.acquire("fake"):
            pass
    elapsed = time.monotonic() - start

    # rate=2 per 0.5s means a new slot opens every 0.25s — the 3rd
    # call should have waited roughly that long, not the full period
    assert elapsed >= 0.2


async def test_unconfigured_provider_raises():
    limiter = RateLimiter()

    with pytest.raises(KeyError):
        async with limiter.acquire("never_configured"):
            pass