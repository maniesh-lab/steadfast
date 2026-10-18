"""
The shapes Steadfast hands back, the same for every provider.

Each provider's SDK answers in its own format. The adapters in
providers/ translate those answers into these small classes, so your
code never has to care which provider replied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """The model asking you to run one of your tools.

    id: the provider's ID for this call. Send it back with the tool's
        result so the model knows which call the result belongs to.
    name: which tool the model wants.
    arguments: the inputs, already parsed from JSON into a dict.
        (None only if the model sent broken JSON. The client retries
        those automatically, so responses it returns never contain one.)
    raw_arguments: the exact text the model sent, before parsing.
    """

    id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str = ""


@dataclass
class Usage:
    """How many tokens a call (or a whole client) used."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


@dataclass
class LLMResponse:
    """One reply from a model, in the same shape for every provider.

    text: the reply's text ("" if the model only called tools).
    tool_calls: tools the model wants you to run (empty list if none).
    finish_reason: why the model stopped. Normalized to "stop",
        "length" (hit max_tokens), or "tool_calls"; anything else is
        passed through as the provider sent it.
    usage: tokens used by this reply.
    model: the model that answered.
    provider: the provider name that answered (useful with fallbacks).
    parsed: the validated Pydantic object, when you passed a schema.
    raw: the provider SDK's original response object, for anything
        Steadfast doesn't expose.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    provider: str | None = None
    parsed: Any = None
    raw: Any = None

    def to_message(self) -> dict[str, Any]:
        """This reply as an assistant message, ready to append to your
        message list for the next turn (including any tool calls)."""
        message: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments},
                }
                for call in self.tool_calls
            ]
        return message


def tool_result(call: ToolCall, content: str) -> dict[str, Any]:
    """Build the message that sends a tool's result back to the model."""
    return {"role": "tool", "tool_call_id": call.id, "content": content}
