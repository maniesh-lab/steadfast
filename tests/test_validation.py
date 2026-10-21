"""
Tests for validation.py: proves the validate-and-retry loop works
without needing any real LLM call.
"""

import pytest
from pydantic import ValidationError

from steadfast.exceptions import ValidationRetryExceeded
from steadfast.validation import (
    build_feedback,
    extract_json,
    format_errors,
    validate_json,
    validate_with_retry,
)


def test_validate_json_success(fact_schema):
    result = validate_json(fact_schema, '{"topic": "octopuses", "fact": "they have three hearts"}')
    assert result.topic == "octopuses"
    assert result.fact == "they have three hearts"


def test_validate_json_failure_raises(fact_schema):
    with pytest.raises(ValidationError):
        validate_json(fact_schema, '{"topic": "octopuses"}')  # missing "fact"


async def test_validate_with_retry_succeeds_after_bad_first_attempt(fact_schema):
    responses = [
        '{"topic": "octopuses"}',  # missing "fact": fails validation
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


# ---------- v0.2: JSON extraction ----------

GOOD = '{"topic": "octopuses", "fact": "they have three hearts"}'


def test_extract_json_strips_code_fence():
    assert extract_json(f"```json\n{GOOD}\n```") == GOOD


def test_extract_json_strips_chatter_around_json():
    assert extract_json(f"Sure! Here you go: {GOOD} Hope that helps.") == GOOD


def test_extract_json_leaves_plain_json_alone():
    assert extract_json(GOOD) == GOOD


def test_extract_json_handles_none():
    assert extract_json(None) == ""


def test_validate_json_accepts_fenced_json(fact_schema):
    result = validate_json(fact_schema, f"```json\n{GOOD}\n```")
    assert result.topic == "octopuses"


# ---------- v0.2: error formatting and feedback ----------


def test_format_errors_names_the_field(fact_schema):
    with pytest.raises(ValidationError) as exc_info:
        validate_json(fact_schema, '{"topic": "octopuses"}')
    problems = format_errors(exc_info.value)
    assert any(p.startswith("fact:") for p in problems)


def test_build_feedback_contains_reply_and_problems():
    feedback = build_feedback('{"topic": "x"}', ["fact: Field required"])
    assert '{"topic": "x"}' in feedback
    assert "- fact: Field required" in feedback


def test_build_feedback_truncates_huge_replies():
    feedback = build_feedback("x" * 10_000, ["bad"])
    assert len(feedback) < 3000


# ---------- v0.2: loop behaviour ----------


async def test_feedback_is_passed_to_next_attempt(fact_schema):
    seen = []
    responses = ['{"topic": "octopuses"}', GOOD]

    async def fake_call(feedback):
        seen.append(feedback)
        return responses[len(seen) - 1]

    await validate_with_retry(fact_schema, fake_call, max_attempts=3)

    assert seen[0] is None  # first attempt: no feedback yet
    assert '{"topic": "octopuses"}' in seen[1]  # model's bad reply echoed
    assert "fact" in seen[1]  # and what was wrong


async def test_fenced_reply_succeeds_without_retry(fact_schema):
    calls = {"n": 0}

    async def fake_call(feedback):
        calls["n"] += 1
        return f"```json\n{GOOD}\n```"

    result = await validate_with_retry(fact_schema, fake_call)

    assert result.topic == "octopuses"
    assert calls["n"] == 1  # no wasted retry


async def test_custom_validator_triggers_retry_then_passes(fact_schema):
    responses = [
        '{"topic": "octopuses", "fact": "TODO"}',
        GOOD,
    ]
    seen = []

    async def fake_call(feedback):
        seen.append(feedback)
        return responses[len(seen) - 1]

    def no_placeholders(result):
        return "fact is a placeholder" if result.fact == "TODO" else None

    result = await validate_with_retry(
        fact_schema, fake_call, max_attempts=3, validators=[no_placeholders]
    )

    assert result.fact == "they have three hearts"
    assert "fact is a placeholder" in seen[1]


async def test_exhaustion_carries_response_and_errors(fact_schema):
    async def always_bad(feedback):
        return '{"topic": "octopuses"}'

    with pytest.raises(ValidationRetryExceeded) as exc_info:
        await validate_with_retry(fact_schema, always_bad, max_attempts=2)

    exc = exc_info.value
    assert exc.attempts == 2
    assert exc.last_response == '{"topic": "octopuses"}'
    assert any("fact" in e for e in exc.errors)
    assert "fact" in exc.last_error


async def test_custom_validator_failure_exhausts_too(fact_schema):
    async def fake_call(feedback):
        return GOOD

    def always_unhappy(result):
        return "never good enough"

    with pytest.raises(ValidationRetryExceeded) as exc_info:
        await validate_with_retry(
            fact_schema, fake_call, max_attempts=2, validators=[always_unhappy]
        )

    assert exc_info.value.errors == ["never good enough"]


def test_exception_new_fields_are_optional():
    exc = ValidationRetryExceeded("boom", attempts=3)
    assert exc.last_response is None
    assert exc.errors == []


def test_exception_errors_lists_are_not_shared():
    a = ValidationRetryExceeded("a", attempts=1)
    b = ValidationRetryExceeded("b", attempts=1)
    a.errors.append("only on a")
    assert b.errors == []
