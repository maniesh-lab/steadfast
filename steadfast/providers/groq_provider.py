"""
Groq adapter, using the official groq SDK.

Groq speaks OpenAI's format, so the work happens in
_openai_compatible.py. The one Groq-specific rule is below.
"""

try:
    import groq
except ImportError as exc:  # pragma: no cover - only without the SDK installed
    raise ImportError(
        'GroqProvider needs the groq SDK: pip install "steadfast-llm[groq]"'
    ) from exc

from ..exceptions import RetryableProviderError
from ._openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    model = "openai/gpt-oss-20b"  # default; pass model=... to use another
    sdk = groq

    def _make_client(self, **options):
        return groq.AsyncGroq(**options)

    def _map_error(self, exc):
        # When a model writes a broken tool call, Groq rejects it with a
        # 400 "tool_use_failed" instead of returning it. Asking again
        # usually works, so treat it as retryable rather than permanent.
        if isinstance(exc, groq.BadRequestError) and "tool_use_failed" in _error_text(exc):
            return RetryableProviderError(
                f"model produced an invalid tool call: {exc}",
                provider=self.name,
                original_exception=exc,
            )
        return super()._map_error(exc)


def _error_text(exc):
    return f"{exc} {getattr(exc, 'body', '')} {getattr(exc, 'code', '')}"
