"""
Anthropic adapter, using the official anthropic SDK.

Steadfast takes messages and tools in OpenAI's format everywhere, so
this file translates both ways:

- requests:  system messages -> the `system` parameter,
             assistant tool_calls -> tool_use blocks,
             role "tool" messages -> tool_result blocks in a user turn
- responses: text blocks -> text, tool_use blocks -> ToolCall
"""

from __future__ import annotations

import json
from typing import Any

try:
    import anthropic
except ImportError as exc:  # pragma: no cover - only without the SDK installed
    raise ImportError(
        'AnthropicProvider needs the anthropic SDK: pip install "steadfast-llm[anthropic]"'
    ) from exc

from ..exceptions import RetryableProviderError
from ..types import LLMResponse, ToolCall, Usage
from .base import DEFAULT_MAX_TOKENS, BaseProvider, status_error

_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def _as_blocks(content: Any) -> list[dict[str, Any]]:
    """Message content as a list of Anthropic content blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        return list(content)  # assume blocks already (text parts match)
    return [{"type": "text", "text": str(content)}]


def _text_of(content: Any) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Translate OpenAI-format messages. Returns (system, messages)."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        # Anthropic needs user/assistant turns to alternate, so merge
        # back-to-back messages from the same role into one turn.
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            system_parts.append(_text_of(content))

        elif role == "tool":
            add(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": _text_of(content),
                    }
                ],
            )

        elif role == "assistant":
            blocks = _as_blocks(content)
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                arguments = function.get("arguments") or "{}"
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        arguments = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": function.get("name"),
                        "input": arguments,
                    }
                )
            add("assistant", blocks)

        else:  # "user" (and anything unknown is treated as user)
            add("user", _as_blocks(content))

    system = "\n\n".join(part for part in system_parts if part) or None
    return system, out


def to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for tool in tools:
        function = tool.get("function", tool)
        converted.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


def to_anthropic_tool_choice(choice: Any) -> dict[str, Any] | None:
    if choice is None:
        return None
    if choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if choice == "none":
        return {"type": "none"}
    if isinstance(choice, dict):
        name = choice.get("function", {}).get("name") or choice.get("name")
        if name:
            return {"type": "tool", "name": name}
        return choice  # already in Anthropic's format
    raise ValueError(f"unknown tool_choice: {choice!r}")


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    model = "claude-sonnet-4-6"  # default; pass model=... to use another

    def __init__(self, api_key=None, model=None, timeout=60.0, **client_options):
        if model is not None:
            self.model = model
        # max_retries=0: Steadfast owns all retry behavior, not the SDK
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key, max_retries=0, timeout=timeout, **client_options
        )

    async def chat(self, messages, tools=None, tool_choice=None, **params):
        system, converted = to_anthropic_messages(messages)

        request: dict[str, Any] = {"model": self.model, "messages": converted, **params}
        if "max_completion_tokens" in request:
            request.setdefault("max_tokens", request.pop("max_completion_tokens"))
        request.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        if "stop" in request:
            stop = request.pop("stop")
            request["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        if system:
            request["system"] = system
        if tools:
            request["tools"] = to_anthropic_tools(tools)
            mapped_choice = to_anthropic_tool_choice(tool_choice)
            if mapped_choice is not None:
                request["tool_choice"] = mapped_choice

        try:
            response = await self.client.messages.create(**request)
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise RetryableProviderError(
                f"connection failed: {exc}", provider=self.name, original_exception=exc
            ) from exc
        except anthropic.APIStatusError as exc:
            raise status_error(self.name, exc, exc.status_code, exc.response.headers) from exc

        return self._parse_response(response)

    def _parse_response(self, response) -> LLMResponse:
        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments),
                    )
                )

        usage = Usage()
        if getattr(response, "usage", None) is not None:
            usage = Usage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
            )

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=_STOP_REASONS.get(response.stop_reason, response.stop_reason),
            usage=usage,
            model=getattr(response, "model", None) or self.model,
            provider=self.name,
            raw=response,
        )
