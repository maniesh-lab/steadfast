"""
Tests for validation.py — proves the validate-and-retry loop works
without needing any real LLM call.
"""
import pytest
from pydantic import ValidationError

from steadfast.exceptions import ValidationRetryExceeded
from steadfast.validation import validate_json, validate_with_retry


def test_validate_json_success(fact_schema):
    result = validate_json(fact_schema, '{"topic": "octopuses", "fact": "they have three hearts"}')
    assert result.topic == "octopuses"
    assert result.fact == "they have three hearts"


def test_validate_json_failure_raises(fact_schema):
    with pytest.raises(ValidationError):
        validate_json(fact_schema, '{"topic": "octopuses"}')  # missing "fact"


async def test_validate_with_retry_succeeds_after_bad_first_attempt(fact_schema):
    responses = [
        '{"topic": "octopuses"}',  # missing "fact" — fails validation
        '{"topic": "octopuses", "fact": "they have three hearts"}',  # valid
    ]
    call_count = {"n": 0}

    async def fake_call(repair_hint):
        raw = responses[call_count["n"]]
        call_count["n"] += 1
        return raw

    result = await validate_with_retry(fact_schema, fake_call, max_attempts=3)

    assert result.fact == "they have three hearts"
    assert call_count["n"] == 2


async def test_validate_with_retry_gives_up(fact_schema):
    async def always_bad(repair_hint):
        return '{"topic": "octopuses"}'  # always missing "fact"

    with pytest.raises(ValidationRetryExceeded) as exc_info:
        await validate_with_retry(fact_schema, always_bad, max_attempts=3)

    assert exc_info.value.attempts == 3