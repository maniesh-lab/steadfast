"""
Tests for the provider adapters. The real SDK clients are swapped for
fakes, so these check Steadfast's own work: building the request,
reading the response, translating formats, and sorting errors into
retryable vs permanent. No API key or network needed.
"""

from types import SimpleNamespace as NS

import anthropic
import groq
import httpx
import openai
import pytest

from steadfast import (
    NonRetryableProviderError,
    RetryableProviderError,
    SteadfastClient,
    tool_result,
)
from steadfast.providers.anthropic_provider import (
    AnthropicProvider,
    to_anthropic_messages,
    to_anthropic_tool_choice,
    to_anthropic_tools,
)
from steadfast.providers.base import parse_retry_after, status_error
from steadfast.providers.groq_provider import GroqProvider
from steadfast.providers.openai_provider import OpenAIProvider

REQUEST = httpx.Request("POST", "https://api.example.test")


def http_response(status, headers=None):
    return httpx.Response(status, headers=headers or {}, request=REQUEST)


class FakeCreate:
    """Stands in for client.chat.completions.create / client.messages.create."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    async def __call__(self, **request):
        self.calls.append(request)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def openai_style(provider_cls, result):
    provider = provider_cls(api_key="test")
    create = FakeCreate(result)
    provider.client = NS(chat=NS(completions=NS(create=create)))
    return provider, create


def anthropic_style(result):
    provider = AnthropicProvider(api_key="test")
    create = FakeCreate(result)
    provider.client = NS(messages=NS(create=create))
    return provider, create


def openai_reply(content="hi", tool_calls=None, finish="stop"):
    message = NS(content=content, tool_calls=tool_calls)
    return NS(
        choices=[NS(message=message, finish_reason=finish)],
        usage=NS(prompt_tokens=12, completion_tokens=7),
        model="served-model",
    )


def openai_tool_call(call_id, name, arguments):
    return NS(id=call_id, function=NS(name=name, arguments=arguments))


# --- shared error rules -----------------------------------------------------


def test_parse_retry_after():
    assert parse_retry_after({"retry-after": "3"}) == 3.0
    assert parse_retry_after({"retry-after-ms": "1500", "retry-after": "9"}) == 1.5
    assert parse_retry_after({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}) is None
    assert parse_retry_after({}) is None
    assert parse_retry_after(None) is None


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 529])
def test_transient_statuses_are_retryable(status):
    assert isinstance(status_error("p", Exception("x"), status), RetryableProviderError)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_permanent_statuses_are_not_retryable(status):
    assert isinstance(status_error("p", Exception("x"), status), NonRetryableProviderError)


# --- OpenAI / Groq ----------------------------------------------------------


async def test_openai_request_is_built_correctly():
    provider, create = openai_style(OpenAIProvider, openai_reply())
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

    await provider.chat(
        [{"role": "user", "content": "q"}], tools=tools, tool_choice="auto", max_tokens=99, temperature=0
    )

    request = create.calls[0]
    assert request["model"] == "gpt-4o"
    assert request["max_completion_tokens"] == 99
    assert "max_tokens" not in request
    assert request["temperature"] == 0
    assert request["tools"] == tools
    assert request["tool_choice"] == "auto"


async def test_openai_defaults_max_tokens_and_skips_empty_tools():
    provider, create = openai_style(OpenAIProvider, openai_reply())

    await provider.chat([{"role": "user", "content": "q"}], tool_choice="auto")

    assert create.calls[0]["max_completion_tokens"] == 1024
    assert "tools" not in create.calls[0]
    assert "tool_choice" not in create.calls[0]


async def test_openai_response_is_parsed():
    calls = [
        openai_tool_call("c1", "search", '{"q": "refund"}'),
        openai_tool_call("c2", "search", '{"q": "ref'),  # cut off -> broken
    ]
    provider, _ = openai_style(OpenAIProvider, openai_reply(None, calls, "tool_calls"))

    response = await provider.chat([{"role": "user", "content": "q"}])

    assert response.text == ""
    assert response.finish_reason == "tool_calls"
    assert response.usage.input_tokens == 12 and response.usage.output_tokens == 7
    assert response.model == "served-model"
    assert response.provider == "openai"
    assert response.tool_calls[0].arguments == {"q": "refund"}
    assert response.tool_calls[1].arguments is None
    assert response.tool_calls[1].raw_arguments == '{"q": "ref'


async def test_openai_non_object_arguments_count_as_broken():
    calls = [openai_tool_call("c1", "search", '["not", "an", "object"]')]
    provider, _ = openai_style(OpenAIProvider, openai_reply(None, calls, "tool_calls"))

    response = await provider.chat([{"role": "user", "content": "q"}])

    assert response.tool_calls[0].arguments is None


async def test_openai_rate_limit_is_retryable_with_retry_after():
    error = openai.RateLimitError("slow down", response=http_response(429, {"retry-after": "3"}), body=None)
    provider, _ = openai_style(OpenAIProvider, error)

    with pytest.raises(RetryableProviderError) as info:
        await provider.chat([{"role": "user", "content": "q"}])

    assert info.value.retry_after == 3.0
    assert info.value.original_exception is error


async def test_openai_timeout_is_retryable():
    provider, _ = openai_style(OpenAIProvider, openai.APITimeoutError(request=REQUEST))

    with pytest.raises(RetryableProviderError):
        await provider.chat([{"role": "user", "content": "q"}])


async def test_openai_bad_key_is_not_retryable():
    error = openai.AuthenticationError("bad key", response=http_response(401), body=None)
    provider, _ = openai_style(OpenAIProvider, error)

    with pytest.raises(NonRetryableProviderError):
        await provider.chat([{"role": "user", "content": "q"}])


async def test_groq_uses_same_format():
    provider, create = openai_style(GroqProvider, openai_reply("hey"))

    response = await provider.chat([{"role": "user", "content": "q"}])

    assert response.text == "hey"
    assert response.provider == "groq"
    assert create.calls[0]["model"] == "openai/gpt-oss-20b"


async def test_groq_tool_use_failed_is_retryable():
    error = groq.BadRequestError(
        "Failed to call a function", response=http_response(400), body={"error": {"code": "tool_use_failed"}}
    )
    provider, _ = openai_style(GroqProvider, error)

    with pytest.raises(RetryableProviderError):
        await provider.chat([{"role": "user", "content": "q"}])


async def test_groq_other_bad_requests_are_not_retryable():
    error = groq.BadRequestError("bad request", response=http_response(400), body=None)
    provider, _ = openai_style(GroqProvider, error)

    with pytest.raises(NonRetryableProviderError):
        await provider.chat([{"role": "user", "content": "q"}])


def test_custom_model_and_timeout_are_used():
    provider = OpenAIProvider(api_key="test", model="gpt-4.1-mini", timeout=5)
    assert provider.model == "gpt-4.1-mini"
    assert provider.client.timeout == 5
    assert provider.client.max_retries == 0


# --- Anthropic translation ---------------------------------------------------


def test_anthropic_message_translation():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "refund policy?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "search", "arguments": '{"q": "refund"}'}},
                {"id": "t2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "30 days"},
        {"role": "tool", "tool_call_id": "t2", "content": "premium"},
        {"role": "user", "content": "thanks"},
    ]

    system, converted = to_anthropic_messages(messages)

    assert system == "Be brief.\n\nUse tools."
    assert [m["role"] for m in converted] == ["user", "assistant", "user"]
    assert converted[1]["content"] == [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "refund"}},
        {"type": "tool_use", "id": "t2", "name": "lookup", "input": {}},
    ]
    # both tool results and the next user message merge into one user turn
    assert converted[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "30 days"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "premium"},
        {"type": "text", "text": "thanks"},
    ]


def test_anthropic_tools_and_tool_choice_translation():
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
    assert to_anthropic_tools(tools) == [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
    assert to_anthropic_tool_choice("auto") == {"type": "auto"}
    assert to_anthropic_tool_choice("required") == {"type": "any"}
    assert to_anthropic_tool_choice("none") == {"type": "none"}
    assert to_anthropic_tool_choice({"type": "function", "function": {"name": "f"}}) == {"type": "tool", "name": "f"}
    assert to_anthropic_tool_choice(None) is None
    with pytest.raises(ValueError):
        to_anthropic_tool_choice("sometimes")


async def test_anthropic_request_and_response():
    result = NS(
        content=[
            NS(type="text", text="Checking."),
            NS(type="tool_use", id="t1", name="search", input={"q": "x"}),
        ],
        stop_reason="tool_use",
        usage=NS(input_tokens=20, output_tokens=8),
        model="claude-x",
    )
    provider, create = anthropic_style(result)
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    response = await provider.chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        tools=tools,
        tool_choice="required",
        max_completion_tokens=50,
        stop="END",
    )

    request = create.calls[0]
    assert request["system"] == "sys"
    assert request["max_tokens"] == 50
    assert request["stop_sequences"] == ["END"]
    assert request["tool_choice"] == {"type": "any"}
    assert request["tools"][0]["name"] == "search"
    assert response.text == "Checking."
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].arguments == {"q": "x"}
    assert response.tool_calls[0].raw_arguments == '{"q": "x"}'
    assert response.usage.total_tokens == 28


async def test_anthropic_max_tokens_stop_reason_is_length():
    result = NS(content=[NS(type="text", text="cut")], stop_reason="max_tokens", usage=None, model=None)
    provider, create = anthropic_style(result)

    response = await provider.chat([{"role": "user", "content": "q"}])

    assert response.finish_reason == "length"
    assert create.calls[0]["max_tokens"] == 1024
    assert response.model == "claude-sonnet-4-6"


async def test_anthropic_overloaded_is_retryable():
    error = anthropic.APIStatusError("overloaded", response=http_response(529), body=None)
    provider, _ = anthropic_style(error)

    with pytest.raises(RetryableProviderError):
        await provider.chat([{"role": "user", "content": "q"}])


async def test_anthropic_connection_error_is_retryable():
    provider, _ = anthropic_style(anthropic.APIConnectionError(request=REQUEST))

    with pytest.raises(RetryableProviderError):
        await provider.chat([{"role": "user", "content": "q"}])


# --- a full tool round trip through the client ---------------------------------


async def test_full_tool_loop_through_the_client():
    """Model asks for a tool, we run it and send the result back, model answers."""
    provider = OpenAIProvider(api_key="test")
    create = FakeCreate(None)
    replies = [
        openai_reply(None, [openai_tool_call("c1", "get_plan", '{"customer_id": 7}')], "tool_calls"),
        openai_reply("Customer 7 is on the premium plan."),
    ]

    async def scripted(**request):
        create.calls.append(request)
        return replies.pop(0)

    provider.client = NS(chat=NS(completions=NS(create=scripted)))
    client = SteadfastClient()
    client.register(provider, rate=1000, period=1)

    messages = [{"role": "user", "content": "What plan is customer 7 on?"}]
    first = await client.chat("openai", messages, tools=[{"type": "function", "function": {"name": "get_plan"}}])
    call = first.tool_calls[0]
    messages += [first.to_message(), tool_result(call, "premium")]
    final = await client.chat("openai", messages)

    assert final.text == "Customer 7 is on the premium plan."
    sent = create.calls[1]["messages"]
    assert sent[1]["tool_calls"][0]["function"] == {"name": "get_plan", "arguments": '{"customer_id": 7}'}
    assert sent[2] == {"role": "tool", "tool_call_id": "c1", "content": "premium"}
    assert client.usage["openai"].input_tokens == 24
