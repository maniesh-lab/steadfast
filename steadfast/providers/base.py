"""
The contract every provider adapter follows, plus the error rules they
share.

This is what makes client.py provider-agnostic: it calls
provider.chat(...) without caring whether "provider" is Anthropic,
OpenAI, or Groq underneath, as long as each one follows this shape.

Messages and tools always come in OpenAI's format (the most common one).
Adapters for other providers translate them.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

from ..exceptions import NonRetryableProviderError, RetryableProviderError
from ..types import LLMResponse

# HTTP status codes worth retrying. Everything else (400 bad request,
# 401 bad key, 403, 404, 422...) fails the same way every time.
#   408 request timeout, 409 conflict, 429 rate limited, 5xx server trouble
#   (529 is Anthropic's "overloaded", which is also >= 500)
RETRYABLE_STATUS_CODES = {408, 409, 429}

DEFAULT_MAX_TOKENS = 1024


def parse_retry_after(headers: Any) -> float | None:
    """Read how long the provider asked us to wait, in seconds.

    Looks at retry-after-ms first (more precise), then retry-after.
    Returns None if neither is there or the value isn't a number
    (retry-after can also be a date, which we just ignore).
    """
    if headers is None:
        return None
    for key, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(key)
        if value is None:
            continue
        try:
            seconds = float(value) / scale
        except (TypeError, ValueError):
            continue
        if seconds >= 0:
            return seconds
    return None


def status_error(provider: str, exc: Exception, status_code: int, headers: Any = None):
    """Turn an SDK's HTTP error into the right Steadfast error, by status code."""
    if status_code in RETRYABLE_STATUS_CODES or status_code >= 500:
        return RetryableProviderError(
            f"HTTP {status_code}: {exc}",
            provider=provider,
            original_exception=exc,
            retry_after=parse_retry_after(headers),
        )
    return NonRetryableProviderError(
        f"HTTP {status_code}: {exc}", provider=provider, original_exception=exc
    )


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Squash a message list into one prompt string, for providers that
    only accept a single prompt. A lone user message stays exactly as it
    is; anything longer gets "role: content" lines."""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return str(messages[0].get("content") or "")
    return "\n\n".join(f"{m.get('role')}: {m.get('content') or ''}" for m in messages)


class BaseProvider(ABC):
    """Subclass this to add a provider.

    Implement chat() (recommended: supports messages and tools). Older,
    simpler providers can implement only complete(prompt) instead: the
    default chat() then squashes the messages into one prompt, without
    tool support.

    Adapters must raise RetryableProviderError or NonRetryableProviderError
    (from steadfast.exceptions) on failure, never the raw SDK error,
    otherwise the retry logic can't tell what to do with it.
    """

    name = "base"  # each subclass overrides this, e.g. "anthropic"
    model: str | None = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **params: Any,
    ) -> LLMResponse:
        """Send a conversation to the model and return its reply."""
        if type(self).complete is BaseProvider.complete:
            raise NotImplementedError(f"{type(self).__name__} must implement chat() or complete()")
        if tools:
            raise NotImplementedError(f"{type(self).__name__} doesn't support tool calling")
        text = await self.complete(flatten_messages(messages), **params)
        return LLMResponse(text=text or "", finish_reason="stop", model=self.model, provider=self.name)

    async def complete(self, prompt: str, **params: Any) -> str:
        """Send one prompt, return the reply's text."""
        if type(self).chat is BaseProvider.chat:
            raise NotImplementedError(f"{type(self).__name__} must implement chat() or complete()")
        response = await self.chat([{"role": "user", "content": prompt}], **params)
        return response.text
