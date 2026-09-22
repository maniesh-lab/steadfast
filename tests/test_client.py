"""
Tests for client.py — the full wiring, tested with a fake provider so
no real API calls happen and no real API key is needed.
"""
import pytest

from steadfast.client import SteadfastClient
from steadfast.exceptions import RetryableProviderError
from steadfast.providers.base import BaseProvider


class FakeProvider(BaseProvider):
    """A stand-in provider — returns pre-scripted responses instead of
    calling a real API. Each call to complete() pops the next one off
    the list, in order."""

    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    async def complete(self, prompt, **kwargs):
        self.call_count += 1
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