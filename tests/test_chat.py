"""
Tests for SteadfastClient.chat() and the features around it: tool calls,
multi-turn feedback, fallbacks, usage tracking, and retry settings.
All against scripted fake providers, so no API key or network is needed.
"""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from steadfast import (
    AllProvidersFailed,
    LLMResponse,
    NonRetryableProviderError,
    RetryableProviderError,
    SteadfastClient,
    ToolCall,
    Usage,
    ValidationRetryExceeded,
    function_tool,
    tool_result,
)
from steadfast.providers.base import BaseProvider
from steadfast.retry import RETRY_AFTER_CAP, _wait_backoff_or_retry_after


class ScriptedProvider(BaseProvider):
    """Returns pre-scripted LLMResponses (or raises exceptions) in order,
    and records every message list it was sent."""

    def __init__(self, name, script):
        self.name = name
        self.script = list(script)
        self.sent = []  # one entry per call: (messages, tools, tool_choice, params)

    async def chat(self, messages, tools=None, tool_choice=None, **params):
        self.sent.append((list(messages), tools, tool_choice, params))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def reply(text="", tool_calls=None, finish="stop", tokens=(10, 5)):
    return LLMResponse(
        text=text,
        tool_calls=tool_calls or [],
        finish_reason=finish,
        usage=Usage(*tokens),
    )


def make_client(*providers, **settings):
    settings.setdefault("min_wait", 0.001)
    settings.setdefault("max_wait", 0.002)
    client = SteadfastClient(**settings)
    for provider in providers:
        client.register(provider, rate=1000, period=1)
    return client


# --- basics ---------------------------------------------------------------


async def test_chat_returns_response_and_sends_params():
    provider = ScriptedProvider("p", [reply("hello")])
    client = make_client(provider)
    messages = [{"role": "user", "content": "hi"}]

    response = await client.chat("p", messages, temperature=0.2, max_tokens=50)

    assert response.text == "hello"
    assert response.provider == "p"
    assert provider.sent[0][3] == {"temperature": 0.2, "max_tokens": 50}


async def test_callers_message_list_is_never_modified(fact_schema):
    provider = ScriptedProvider("p", [reply('{"topic": "x"}'), reply('{"topic": "x", "fact": "y"}')])
    client = make_client(provider)
    messages = [{"role": "user", "content": "give me a fact"}]

    await client.chat("p", messages, schema=fact_schema)

    assert messages == [{"role": "user", "content": "give me a fact"}]


async def test_usage_is_tracked_per_provider_and_in_total():
    a = ScriptedProvider("a", [reply("1", tokens=(10, 2)), reply("2", tokens=(5, 1))])
    b = ScriptedProvider("b", [reply("3", tokens=(100, 50))])
    client = make_client(a, b)

    await client.complete("a", "x")
    await client.complete("a", "x")
    await client.complete("b", "x")

    assert client.usage["a"] == Usage(15, 3)
    assert client.usage["b"] == Usage(100, 50)
    assert client.total_usage.total_tokens == 168


async def test_complete_puts_system_prompt_first():
    provider = ScriptedProvider("p", [reply("ok")])
    client = make_client(provider)

    await client.complete("p", "question", system="be brief")

    assert provider.sent[0][0] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "question"},
    ]


async def test_bad_settings_are_rejected():
    with pytest.raises(ValueError):
        SteadfastClient(max_retries=0)
    client = make_client(ScriptedProvider("p", []))
    with pytest.raises(ValueError):
        await client.chat("p", [{"role": "user", "content": "x"}], max_attempts=0)
    with pytest.raises(KeyError):
        await client.chat("p", [{"role": "user", "content": "x"}], fallbacks=["missing"])


# --- validation feedback --------------------------------------------------


async def test_schema_feedback_is_sent_as_conversation_turns(fact_schema):
    provider = ScriptedProvider("p", [reply('{"topic": "x"}'), reply('{"topic": "x", "fact": "y"}')])
    client = make_client(provider)

    response = await client.chat("p", [{"role": "user", "content": "fact?"}], schema=fact_schema)

    assert (response.parsed.topic, response.parsed.fact) == ("x", "y")
    second_call = provider.sent[1][0]
    assert second_call[1] == {"role": "assistant", "content": '{"topic": "x"}'}
    assert second_call[2]["role"] == "user"
    assert "fact: Field required" in second_call[2]["content"]


async def test_cut_off_reply_gets_a_shorter_answer_hint(fact_schema):
    provider = ScriptedProvider(
        "p", [reply('{"topic": "x", "fa', finish="length"), reply('{"topic": "x", "fact": "y"}')]
    )
    client = make_client(provider)

    await client.chat("p", [{"role": "user", "content": "fact?"}], schema=fact_schema)

    assert "token limit" in provider.sent[1][0][-1]["content"]


async def test_async_validators_are_awaited(fact_schema):
    async def no_todo(result):
        return "placeholder fact" if result.fact == "TODO" else None

    provider = ScriptedProvider(
        "p", [reply('{"topic": "x", "fact": "TODO"}'), reply('{"topic": "x", "fact": "real"}')]
    )
    client = make_client(provider)

    response = await client.chat(
        "p", [{"role": "user", "content": "fact?"}], schema=fact_schema, validators=[no_todo]
    )

    assert response.parsed.fact == "real"


# --- tool calls -------------------------------------------------------------


async def test_tool_calls_come_back_and_skip_the_schema(fact_schema):
    call = ToolCall(id="c1", name="search", arguments={"q": "refund"}, raw_arguments='{"q": "refund"}')
    provider = ScriptedProvider("p", [reply(tool_calls=[call], finish="tool_calls")])
    client = make_client(provider)
    tools = [function_tool("search", "Search docs.", {"type": "object", "properties": {}})]

    response = await client.chat(
        "p", [{"role": "user", "content": "q"}], tools=tools, tool_choice="auto", schema=fact_schema
    )

    assert response.tool_calls == [call]
    assert response.parsed is None
    assert provider.sent[0][1] == tools
    assert provider.sent[0][2] == "auto"


async def test_broken_tool_arguments_are_retried_with_feedback():
    broken = ToolCall(id="c1", name="search", arguments=None, raw_arguments='{"q": "refund"')
    fixed = ToolCall(id="c2", name="search", arguments={"q": "refund"}, raw_arguments='{"q": "refund"}')
    provider = ScriptedProvider("p", [reply(tool_calls=[broken]), reply(tool_calls=[fixed])])
    client = make_client(provider)

    response = await client.chat("p", [{"role": "user", "content": "q"}], tools=[{}])

    assert response.tool_calls == [fixed]
    feedback = provider.sent[1][0][-1]
    assert feedback["role"] == "user"
    assert "not valid JSON" in feedback["content"]
    # the broken call is not echoed back as an assistant turn
    assert all(m["role"] != "assistant" for m in provider.sent[1][0])


async def test_broken_tool_arguments_can_exhaust_attempts():
    broken = ToolCall(id="c1", name="search", arguments=None, raw_arguments="{oops")
    provider = ScriptedProvider("p", [reply(tool_calls=[broken])] * 2)
    client = make_client(provider)

    with pytest.raises(ValidationRetryExceeded) as info:
        await client.chat("p", [{"role": "user", "content": "q"}], tools=[{}], max_attempts=2)

    assert info.value.last_response == "{oops"


def test_tool_round_trip_helpers():
    call = ToolCall(id="c1", name="search", arguments={"q": "x"}, raw_arguments='{"q": "x"}')
    response = LLMResponse(text="", tool_calls=[call])

    assert response.to_message() == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}
        ],
    }
    assert tool_result(call, "found 3") == {"role": "tool", "tool_call_id": "c1", "content": "found 3"}


def test_function_tool_accepts_pydantic_dict_or_nothing():
    class Args(BaseModel):
        query: str

    from_model = function_tool("search", "Search.", Args)
    assert from_model["function"]["parameters"]["required"] == ["query"]
    assert function_tool("ping", "Ping.")["function"]["parameters"] == {"type": "object", "properties": {}}
    assert function_tool("x", "X.", {"type": "object"})["function"]["parameters"] == {"type": "object"}
    with pytest.raises(TypeError):
        function_tool("x", "X.", 42)


# --- fallbacks and retries --------------------------------------------------


async def test_falls_back_to_next_provider():
    primary = ScriptedProvider("primary", [NonRetryableProviderError("bad key", provider="primary")])
    backup = ScriptedProvider("backup", [reply("from backup")])
    client = make_client(primary, backup)

    response = await client.chat("primary", [{"role": "user", "content": "q"}], fallbacks=["backup"])

    assert response.text == "from backup"
    assert response.provider == "backup"


async def test_all_providers_failing_lists_every_error():
    a = ScriptedProvider("a", [NonRetryableProviderError("a down", provider="a")])
    b = ScriptedProvider("b", [NonRetryableProviderError("b down", provider="b")])
    client = make_client(a, b)

    with pytest.raises(AllProvidersFailed) as info:
        await client.complete("a", "q", fallbacks=["b"])

    assert [name for name, _ in info.value.errors] == ["a", "b"]


async def test_without_fallbacks_the_real_error_is_raised():
    provider = ScriptedProvider("p", [NonRetryableProviderError("bad key", provider="p")])
    client = make_client(provider)

    with pytest.raises(NonRetryableProviderError):
        await client.complete("p", "q")


async def test_max_retries_setting_is_used():
    errors = [RetryableProviderError("busy", provider="p")] * 3
    provider = ScriptedProvider("p", errors)
    client = make_client(provider, max_retries=2)

    with pytest.raises(RetryableProviderError):
        await client.complete("p", "q")

    assert len(provider.sent) == 2


def _retry_state(exc, attempt=1):
    return SimpleNamespace(attempt_number=attempt, outcome=SimpleNamespace(exception=lambda: exc))


def test_wait_respects_retry_after():
    wait = _wait_backoff_or_retry_after(min_wait=0.01, max_wait=0.02)
    exc = RetryableProviderError("slow down", provider="p", retry_after=5)
    assert wait(_retry_state(exc)) >= 5


def test_wait_caps_huge_retry_after():
    wait = _wait_backoff_or_retry_after(min_wait=0.01, max_wait=0.02)
    exc = RetryableProviderError("slow down", provider="p", retry_after=3600)
    assert wait(_retry_state(exc)) <= RETRY_AFTER_CAP


async def test_same_provider_can_be_registered_twice_under_different_names():
    big = ScriptedProvider("groq", [NonRetryableProviderError("model gone", provider="groq")])
    small = ScriptedProvider("groq", [reply("from small")])
    client = make_client()
    assert client.register(big, name="groq-big") == "groq-big"
    client.register(small, name="groq-small", rate=1000, period=1)

    response = await client.chat("groq-big", [{"role": "user", "content": "q"}], fallbacks=["groq-small"])

    assert response.text == "from small"
    assert response.provider == "groq-small"
    assert client.usage["groq-small"].total_tokens == 15


# --- old-style providers ------------------------------------------------------


async def test_provider_without_chat_still_works():
    class OldStyle:  # doesn't even subclass BaseProvider
        name = "old"

        async def complete(self, prompt, **kwargs):
            return f"echo: {prompt}"

    client = make_client(OldStyle())

    assert await client.complete("old", "hi") == "echo: hi"


async def test_old_style_provider_rejects_tools():
    class OldStyle(BaseProvider):
        name = "old"

        async def complete(self, prompt, **kwargs):
            return "x"

    client = make_client(OldStyle())

    with pytest.raises(NotImplementedError):
        await client.chat("old", [{"role": "user", "content": "q"}], tools=[{}])


async def test_provider_with_neither_method_fails_clearly():
    class Empty(BaseProvider):
        name = "empty"

    with pytest.raises(NotImplementedError):
        await Empty().chat([{"role": "user", "content": "q"}])
