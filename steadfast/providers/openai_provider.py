"""
OpenAI adapter — implements BaseProvider using the official openai SDK.
Same shape as anthropic_provider.py — OpenAI's SDK uses matching
exception names.
"""

import openai
from ..exceptions import NonRetryableProviderError, RetryableProviderError
from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key=None, model="gpt-4o"):
        self.client = openai.AsyncOpenAI(api_key=api_key, max_retries=0)
        self.model = model

    async def complete(self, prompt, **kwargs):
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        except openai.RateLimitError as e:
            raise RetryableProviderError("rate limited", provider=self.name, original_exception=e)
        except openai.APIConnectionError as e:
            raise RetryableProviderError("connection failed", provider=self.name, original_exception=e)
        except openai.InternalServerError as e:
            raise RetryableProviderError("server error", provider=self.name, original_exception=e)
        except openai.APIStatusError as e:
            raise NonRetryableProviderError(str(e), provider=self.name, original_exception=e)