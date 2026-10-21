"""
Shared code for providers that speak OpenAI's chat-completions format.

OpenAI and Groq use the same request and response shape (Groq's SDK
copies OpenAI's), so both adapters are thin subclasses of this one.
"""

from __future__ import annotations

import json
from typing import Any

from ..exceptions import RetryableProviderError
from ..types import LLMResponse, ToolCall, Usage
from .base import DEFAULT_MAX_TOKENS, BaseProvider, status_error


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Parse a tool call's JSON arguments. Raises ValueError if they
    aren't a JSON object, so the client can ask the model to fix them."""
    if not raw or not raw.strip():
        return {}
    value = json.loads(raw)  # json.JSONDecodeError is a ValueError
    if not isinstance(value, dict):
        # ValueError on purpose: the caller treats every bad-arguments case the same
        raise ValueError(f"arguments must be a JSON object, got {type(value).__name__}")  # noqa: TRY004
    return value


class OpenAICompatibleProvider(BaseProvider):
    sdk: Any = None  # the SDK module (openai or groq), set by subclasses

    def __init__(self, api_key=None, model=None, timeout=60.0, **client_options):
        """
        api_key: falls back to the SDK's usual environment variable.
        timeout: seconds before a request counts as failed (and is retried).
        client_options: passed straight to the SDK client (e.g. base_url).
        """
        if model is not None:
            self.model = model
        # max_retries=0: Steadfast owns all retry behavior, not the SDK
        self.client = self._make_client(
            api_key=api_key, max_retries=0, timeout=timeout, **client_options
        )

    def _make_client(self, **options):
        raise NotImplementedError

    def _build_request(self, messages, tools, tool_choice, params):
        request = {"model": self.model, "messages": messages, **params}
        if "max_tokens" in request:
            # Newer models reject max_tokens; max_completion_tokens works everywhere.
            request.setdefault("max_completion_tokens", request.pop("max_tokens"))
        request.setdefault("max_completion_tokens", DEFAULT_MAX_TOKENS)
        if tools:
            request["tools"] = tools
            if tool_choice is not None:
                request["tool_choice"] = tool_choice
        return request

    def _map_error(self, exc):
        """Turn an SDK exception into a Steadfast one, or return None."""
        sdk = self.sdk
        if isinstance(exc, sdk.APIConnectionError):  # includes timeouts
            return RetryableProviderError(
                f"connection failed: {exc}", provider=self.name, original_exception=exc
            )
        if isinstance(exc, sdk.APIStatusError):
            return status_error(self.name, exc, exc.status_code, exc.response.headers)
        return None

    async def chat(self, messages, tools=None, tool_choice=None, **params):
        request = self._build_request(messages, tools, tool_choice, params)
        try:
            response = await self.client.chat.completions.create(**request)
        except Exception as exc:
            mapped = self._map_error(exc)
            if mapped is None:
                raise
            raise mapped from exc
        return self._parse_response(response)

    def _parse_response(self, response) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        for call in message.tool_calls or []:
            raw = call.function.arguments or ""
            try:
                arguments = parse_arguments(raw)
            except ValueError:
                arguments = None  # the client spots this and asks for a fix
            tool_calls.append(
                ToolCall(
                    id=call.id, name=call.function.name, arguments=arguments, raw_arguments=raw
                )
            )

        usage = Usage()
        if getattr(response, "usage", None) is not None:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )

        finish = choice.finish_reason
        if finish == "function_call":
            finish = "tool_calls"

        return LLMResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=usage,
            model=getattr(response, "model", None) or self.model,
            provider=self.name,
            raw=response,
        )
