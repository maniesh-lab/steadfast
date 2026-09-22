"""
Groq adapter — implements BaseProvider using the official groq SDK.
Same shape again — groq's SDK mirrors the same exception names too.
"""

import groq
from ..exceptions import NonRetryableProviderError, RetryableProviderError
from .base import BaseProvider


class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, api_key=None, model="openai/gpt-oss-20b"):
        self.client = groq.AsyncGroq(api_key=api_key, max_retries=0)
        self.model = model

    async def complete(self, prompt, **kwargs):
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        except groq.RateLimitError as e:
            raise RetryableProviderError("rate limited", provider=self.name, original_exception=e)
        except groq.APIConnectionError as e:
            raise RetryableProviderError("connection failed", provider=self.name, original_exception=e)
        except groq.InternalServerError as e:
            raise RetryableProviderError("server error", provider=self.name, original_exception=e)
        except groq.APIStatusError as e:
            raise NonRetryableProviderError(str(e), provider=self.name, original_exception=e)