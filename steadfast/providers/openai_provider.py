"""
OpenAI adapter, using the official openai SDK.

All the work happens in _openai_compatible.py; this file only says
which SDK and which default model to use.
"""

try:
    import openai
except ImportError as exc:  # pragma: no cover - only without the SDK installed
    raise ImportError(
        'OpenAIProvider needs the openai SDK: pip install "steadfast-llm[openai]"'
    ) from exc

from ._openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    model = "gpt-4o"  # default; pass model=... to use another
    sdk = openai

    def _make_client(self, **options):
        return openai.AsyncOpenAI(**options)
