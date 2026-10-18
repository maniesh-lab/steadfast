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


    async def complete(
        self,
        provider_name,
        prompt,
        schema=None,
        max_attempts=3,
        validators=None,
        correlation_id=None,
        **kwargs,
    ):
        """
        Call a registered provider, with rate limiting and retries.
        If schema is given, the response is validated and retried with
        feedback until it matches (or max_attempts runs out).

        max_attempts: total content-validation tries, including the first.
        validators: optional list of functions. Each takes the parsed
            object and returns None if fine, or a string describing the
            problem. Requires schema.
        correlation_id: optional ID to tag this call's log lines with.
            If not given, an ID already bound by the caller (for example
            a web request's ID) is reused, otherwise a new one is made.
        """
        if validators and schema is None:
            raise ValueError("validators need a schema - pass schema=... too.")

        if provider_name not in self.providers:
            raise KeyError(f"Provider '{provider_name}' not registered — call register() first.")

        provider = self.providers[provider_name]


        @with_retry()
        async def call_once(feedback=None):
            full_prompt = prompt
            if feedback:
                full_prompt += (
                    f"\n\n{feedback}\n\n"
                    f"Please fix these problems and respond again "
                    f"with only the corrected JSON."
                )
            async with self.limiter.acquire(provider_name):
                return await provider.complete(full_prompt, **kwargs)

        with correlation_context(correlation_id) as cid:
            log.info("calling provider", provider=provider_name, correlation_id=cid)
            if schema is not None:
                return await validate_with_retry(
                    schema, call_once, max_attempts=max_attempts, validators=validators
                )
            return await call_once()
