"""
OpenAI adapter, using the official openai SDK.

All the work happens in _openai_compatible.py; this file only says
which SDK and which default model to use.
"""

import openai

from ._openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    model = "gpt-4o"  # default; pass model=... to use another
    sdk = openai

    def _make_client(self, **options):
        return openai.AsyncOpenAI(**options)
