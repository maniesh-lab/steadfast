"""
The main entry point for Steadfast. Wires together rate limiting,
retries, validation, and logging around a provider's complete() call.
"""

from .logging_utils import correlation_context, get_logger
from .rate_limit import RateLimiter
from .retry import with_retry
from .validation import validate_with_retry

log = get_logger()


class SteadfastClient:
    def __init__(self):
        self.providers = {}
        self.limiter = RateLimiter()


    def register(self, provider, rate=50, period=60):
        """Register a provider instance and set its rate limit."""
        self.providers[provider.name] = provider
        self.limiter.configure(provider.name, rate=rate, period=period)


    async def complete(self, provider_name, prompt, schema=None, **kwargs):
        """
        Call a registered provider, with rate limiting and retries.
        If schema is given, the response is validated and retried with
        a self-correction hint until it matches (or attempts run out).
        """
        if provider_name not in self.providers:
            raise KeyError(f"Provider '{provider_name}' not registered — call register() first.")

        provider = self.providers[provider_name]


        @with_retry()
        async def call_once(repair_hint=None):
            full_prompt = prompt
            if repair_hint:
                full_prompt += (
                    f"\n\nYour last response had an error: {repair_hint}\n"
                    f"Please fix it and respond again."
                )
            async with self.limiter.acquire(provider_name):
                return await provider.complete(full_prompt, **kwargs)

        with correlation_context() as cid:
            log.info("calling provider", provider=provider_name, correlation_id=cid)
            if schema is not None:
                return await validate_with_retry(schema, call_once)
            return await call_once()