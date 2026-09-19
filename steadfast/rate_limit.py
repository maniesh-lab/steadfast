"""
Rate limiting for Steadfast.

Each provider gets its own independent limiter — Anthropic, OpenAI, and
Groq each set their own separate rate limits, so sharing one limiter
across all three would either throttle a provider that isn't near its
limit, or under-throttle one that is.
"""

import time
from contextlib import asynccontextmanager
from aiolimiter import AsyncLimiter
from .logging_utils import get_logger

log = get_logger()


class RateLimiter:
    """
    Wraps aiolimiter.AsyncLimiter with per-provider limiters and logging.

    Usage:
        limiter = RateLimiter()
        limiter.configure("anthropic", rate=50, period=60)  # 50 calls / 60s

        async with limiter.acquire("anthropic"):
            response = await call_anthropic(...)
    """

    def __init__(self):
        self._limiters: dict[str, AsyncLimiter] = {}


    def configure(self, provider: str, rate: int, period: float = 60.0) -> None:
        """
        Set (or replace) the rate limit for a given provider.
        rate: max number of calls allowed per `period` seconds.
        """

        self._limiters[provider] = AsyncLimiter(rate, period)
        log.info("rate limit configured", provider=provider, rate=rate, period=period)


    @asynccontextmanager
    async def acquire(self, provider: str):
        """
        Waits (if needed) until a call slot is free for this provider,
        then lets the call through. Logs if the wait was meaningful.

        Raises KeyError if configure() hasn't been called for this
        provider yet — fail loudly rather than silently skip the limit.
        """

        if provider not in self._limiters:
            raise KeyError(
                f"No rate limit configured for provider '{provider}' — "
                f"call configure() first."
            )
        
        limiter = self._limiters[provider]
        
        start = time.monotonic()

        async with limiter:
            waited = time.monotonic() - start
            if waited > 0.05:
                log.info("rate limit delayed call", provider=provider, waited_seconds=round(waited, 2))
            yield