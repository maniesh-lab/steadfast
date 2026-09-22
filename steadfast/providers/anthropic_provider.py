"""
Anthropic adapter — implements BaseProvider using the official
anthropic SDK.
"""

import anthropic
from ..exceptions import NonRetryableProviderError, RetryableProviderError
from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key=None, model="claude-sonnet-4-6"):
        # max_retries=0: retry.py owns all retry behavior, not the SDK
        self.client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)
        self.model = model

    async def complete(self, prompt, **kwargs):
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except anthropic.RateLimitError as e:
            raise RetryableProviderError("rate limited", provider=self.name, original_exception=e)
        except anthropic.APIConnectionError as e:
            raise RetryableProviderError("connection failed", provider=self.name, original_exception=e)
        except anthropic.InternalServerError as e:
            raise RetryableProviderError("server error", provider=self.name, original_exception=e)
        except anthropic.APIStatusError as e:
            # bad API key, malformed request, etc. — retrying won't help
            raise NonRetryableProviderError(str(e), provider=self.name, original_exception=e)