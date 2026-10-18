"""
The main entry point for Steadfast.

One call goes through these layers, outside to inside:

    correlation ID      every log line from this call shares one ID
    validation loop     bad output -> feedback to the model -> try again
    fallback chain      a provider that keeps failing -> next provider
    transport retry     rate limit / 5xx / timeout -> wait, try again
    rate limiter        wait for a free slot under the provider's limit
    provider adapter    the actual SDK call
"""

from __future__ import annotations

import time
from typing import Any

from .exceptions import AllProvidersFailed, ProviderError, ValidationRetryExceeded
from .logging_utils import correlation_context, get_logger
from .providers.base import flatten_messages
from .rate_limit import RateLimiter
from .retry import retrying
from .types import LLMResponse, Usage
from .validation import check_response, feedback_messages

log = get_logger()


class SteadfastClient:
    def __init__(self, max_retries: int = 5, min_wait: float = 1.0, max_wait: float = 30.0):
        """
        max_retries: tries per provider for transient errors (rate limits,
            5xx, timeouts), counting the first try.
        min_wait / max_wait: backoff between those tries, in seconds.
        """
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.providers: dict[str, Any] = {}
        self.limiter = RateLimiter()
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.usage: dict[str, Usage] = {}  # tokens used, per provider name

    def register(self, provider, rate: int = 50, period: float = 60) -> None:
        """Register a provider instance and set its rate limit
        (`rate` calls per `period` seconds)."""
        self.providers[provider.name] = provider
        self.limiter.configure(provider.name, rate=rate, period=period)
        self.usage.setdefault(provider.name, Usage())

    @property
    def total_usage(self) -> Usage:
        """Tokens used across every provider since this client was made."""
        total = Usage()
        for usage in self.usage.values():
            total.add(usage)
        return total

    async def chat(
        self,
        provider_name: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        schema=None,
        validators=None,
        max_attempts: int = 3,
        fallbacks: list[str] | None = None,
        correlation_id: str | None = None,
        **params: Any,
    ) -> LLMResponse:
        """
        Send a conversation and get one reply back, as an LLMResponse.

        messages: OpenAI-format messages, e.g.
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            Your list is never modified.
        tools / tool_choice: OpenAI-format tool definitions (see
            steadfast.tools.function_tool). Calls the model makes come
            back in response.tool_calls.
        schema: a Pydantic model. Text replies are validated against it
            and the result is put in response.parsed. Skipped when the
            model calls tools instead of answering.
        validators: extra checks on the parsed object (sync or async);
            each returns None if fine or a string saying what's wrong.
            Requires schema.
        max_attempts: total tries for bad output (invalid JSON, wrong
            shape, failed validator, broken tool-call JSON), first try
            included.
        fallbacks: provider names to try, in order, if the main one still
            fails after its retries.
        correlation_id: ID for this call's log lines. Defaults to one
            already bound by the caller, otherwise a new one.
        params: passed to the provider (max_tokens, temperature, ...).
        """
        if validators and schema is None:
            raise ValueError("validators need a schema - pass schema=... too.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        names = [provider_name, *(fallbacks or [])]
        for name in names:
            if name not in self.providers:
                raise KeyError(f"Provider '{name}' not registered — call register() first.")

        with correlation_context(correlation_id) as cid:
            log.info("calling provider", provider=provider_name, correlation_id=cid)

            conversation = list(messages)
            response = None
            problems: list[str] = []

            for attempt in range(1, max_attempts + 1):
                response = await self._call_with_fallback(
                    names, conversation, tools, tool_choice, params
                )
                parsed, problems = await check_response(response, schema, validators)
                if not problems:
                    response.parsed = parsed
                    return response

                log.warning(
                    "response failed validation",
                    provider=response.provider,
                    attempt=attempt,
                    errors=problems,
                )
                conversation = conversation + feedback_messages(response, problems)

            last = response.text or "\n".join(c.raw_arguments for c in response.tool_calls)
            raise ValidationRetryExceeded(
                f"response failed validation after {max_attempts} attempts",
                attempts=max_attempts,
                last_error="; ".join(problems),
                last_response=last,
                errors=problems,
            )

    async def complete(
        self,
        provider_name: str,
        prompt: str,
        schema=None,
        max_attempts: int = 3,
        validators=None,
        correlation_id: str | None = None,
        *,
        system: str | None = None,
        fallbacks: list[str] | None = None,
        **params: Any,
    ):
        """
        The simple version of chat(): one prompt in, the answer out.

        Returns the reply's text, or the validated Pydantic object if a
        schema is given. system: an optional system prompt.
        """
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        response = await self.chat(
            provider_name,
            messages,
            schema=schema,
            validators=validators,
            max_attempts=max_attempts,
            fallbacks=fallbacks,
            correlation_id=correlation_id,
            **params,
        )
        return response.parsed if schema is not None else response.text

    async def _call_with_fallback(self, names, messages, tools, tool_choice, params) -> LLMResponse:
        errors: list[tuple[str, Exception]] = []
        for index, name in enumerate(names):
            try:
                return await self._call_provider(name, messages, tools, tool_choice, params)
            except ProviderError as exc:
                errors.append((name, exc))
                if index + 1 < len(names):
                    log.warning(
                        "provider failed, falling back",
                        provider=name,
                        next_provider=names[index + 1],
                        error=str(exc),
                    )
        if len(errors) == 1:
            raise errors[0][1]  # no fallbacks: raise the real error
        raise AllProvidersFailed(errors)

    async def _call_provider(self, name, messages, tools, tool_choice, params) -> LLMResponse:
        provider = self.providers[name]
        response = None
        start = time.monotonic()

        async for attempt in retrying(self.max_retries, self.min_wait, self.max_wait):
            with attempt:
                async with self.limiter.acquire(name):
                    response = await _send(provider, messages, tools, tool_choice, params)

        if response.provider is None:
            response.provider = name
        self.usage[name].add(response.usage)
        log.info(
            "provider call finished",
            provider=name,
            model=response.model,
            latency_ms=round((time.monotonic() - start) * 1000),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.finish_reason,
            tool_calls=len(response.tool_calls),
        )
        return response


async def _send(provider, messages, tools, tool_choice, params) -> LLMResponse:
    """Call the provider. Also supports old-style providers that only
    have complete(prompt) and don't subclass BaseProvider."""
    if hasattr(provider, "chat"):
        return await provider.chat(messages, tools=tools, tool_choice=tool_choice, **params)
    if tools:
        raise NotImplementedError(f"provider '{provider.name}' doesn't support tool calling")
    text = await provider.complete(flatten_messages(messages), **params)
    return LLMResponse(text=text or "", finish_reason="stop", provider=provider.name)
