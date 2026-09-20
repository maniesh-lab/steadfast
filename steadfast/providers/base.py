"""
The contract every provider adapter must follow.

This is what makes client.py provider-agnostic: it calls
provider.complete(...) without caring whether "provider" is Anthropic,
OpenAI, or Groq underneath — as long as each one follows this same shape.
"""
from abc import ABC, abstractmethod


class BaseProvider(ABC):
    name = "base"  # each subclass overrides this, e.g. "anthropic"

    @abstractmethod
    async def complete(self, prompt, **kwargs):
        """
        Send prompt to the provider's API, return the raw text response.

        Must raise RetryableProviderError or NonRetryableProviderError
        (from steadfast.exceptions) on failure, not the raw SDK error —
        otherwise retry.py won't know what to do with it.
        """
        raise NotImplementedError