"""
Tests for client.py — the full wiring, tested with a fake provider so
no real API calls happen and no real API key is needed.
"""
import pytest
import structlog

from steadfast.client import SteadfastClient
from steadfast.exceptions import RetryableProviderError, ValidationRetryExceeded
from steadfast.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    """A stand-in provider — returns pre-scripted responses instead of
    calling a real API. Each call to complete() pops the next one off
    the list, in order."""

    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0
        self.prompts = []  # every prompt this provider received

    async def complete(self, prompt, **kwargs):
        self.call_count += 1
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


async def test_plain_call_returns_text():
    provider = FakeProvider(["hello there"])
    client = SteadfastClient()
    client.register(provider, rate=100, period=1)

    result = await client.complete("fake", "say hi")

    assert result == "hello there"


async def test_schema_call_validates_and_returns_object(fact_schema):
    provider = FakeProvider(['{"topic": "x", "fact": "y"}'])
    client = SteadfastClient()
    client.register(provider, rate=100, period=1)

    result = await client.complete("fake", "give me a fact", schema=fact_schema)

    assert result.topic == "x"
    assert result.fact == "y"


async def test_schema_call_retries_after_bad_json(fact_schema):
    provider = FakeProvider([
        '{"topic": "x"}',                    # missing "fact" — fails validation
        '{"topic": "x", "fact": "y"}',        # valid on 2nd try
    ])
    client = SteadfastClient()
    client.register(provider, rate=100, period=1)

    result = await client.complete("fake", "give me a fact", schema=fact_schema)

    assert result.fact == "y"
    assert provider.call_count == 2


async def test_retryable_provider_error_gets_retried():
    # Note: client.py uses @with_retry() with default wait times, so
    # this test takes a real 1-2 seconds — that's expected, not a bug.
    provider = FakeProvider([
        RetryableProviderError("temporary", provider="fake"),
        "recovered",
    ])
    client = SteadfastClient()
    client.register(provider, rate=100, period=1)

    result = await client.complete("fake", "say hi")

    assert result == "recovered"
    assert provider.call_count == 2


async def test_unregistered_provider_raises():
    client = SteadfastClient()
    with pytest.raises(KeyError):
        await client.complete("nonexistent", "say hi")


async def test_feedback_reaches_the_prompt(fact_schema):
    provider = FakeProvider([
        '{"topic": "x"}',                    # missing "fact"
        '{"topic": "x", "fact": "y"}',
    ])
    client = SteadfastClient()
    client.register(provider, rate=100, period=1)

    await client.complete("fake", "give me a fact", schema=fact_schema)

    assert provider.prompts[0] == "give me a fact"  # first prompt untouched
    assert '{"topic": "x"}' in provider.prompts[1]  # model's bad reply echoed
    assert "fact" in provider.prompts[1]            # and what was wrong


async def test_max_attempts_is_respected(fact_schema):
    provider = FakeProvider(['{"topic": "x"}'] * 5)
    client = SteadfastClient()
    client.register(provider, rate=100, period=1)

    with pytest.raises(ValidationRetryExceeded):
        await client.complete("fake", "give me a fact", schema=fact_schema, max_attempts=2)

    assert provider.call_count == 2


async def test_validators_work_through_the_client(fact_schema):
    provider = FakeProvider([
        '{"topic": "x", "fact": "TODO"}',
        '{"topic": "x", "fact": "real fact"}',
    ])
    client = SteadfastClient()
    client.register(provider, rate=100, period=1)

    result = await client.complete(
        "fake",
        "give me a fact",
        schema=fact_schema,
        validators=[lambda r: "placeholder" if r.fact == "TODO" else None],
    )

    assert result.fact == "real fact"
    assert provider.call_count == 2


async def test_validators_without_schema_raises():
    client = SteadfastClient()
    client.register(FakeProvider(["hi"]), rate=100, period=1)

    with pytest.raises(ValueError):
        await client.complete("fake", "say hi", validators=[lambda r: None])


async def test_correlation_id_param_is_used_and_outer_id_survives():
    seen = []

    class RecordingProvider(FakeProvider):
        async def complete(self, prompt, **kwargs):
            seen.append(structlog.contextvars.get_contextvars().get("correlation_id"))
            return await super().complete(prompt, **kwargs)

    client = SteadfastClient()
    client.register(RecordingProvider(["hi"]), rate=100, period=1)

    structlog.contextvars.bind_contextvars(correlation_id="request-id")
    try:
        await client.complete("fake", "say hi", correlation_id="call-id")
        assert seen == ["call-id"]
        assert structlog.contextvars.get_contextvars()["correlation_id"] == "request-id"
    finally:
        structlog.contextvars.unbind_contextvars("correlation_id")
