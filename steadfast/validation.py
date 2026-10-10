"""
Response validation for Steadfast.

Turns raw LLM text into a validated Pydantic object. Three kinds of
failure are caught, and all of them are fed back to the model so it can
fix its own mistake:

1. Not parseable  - no valid JSON in the reply
2. Wrong shape    - valid JSON, but it doesn't match the schema
3. Failed a check - matches the schema, but a custom validator rejected it
"""

import re

from pydantic import ValidationError

from .exceptions import ValidationRetryExceeded
from .logging_utils import get_logger

log = get_logger()

# Matches ```json ... ``` (or plain ``` ... ```) code fences.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Don't paste an enormous bad reply back into the next prompt.
_MAX_ECHO_CHARS = 2000


def extract_json(raw):
    """Pull the JSON out of a raw reply: strips ```json fences and any
    chatter before/after the JSON. If no JSON is found, returns the text
    unchanged so validation fails with a clear error."""
    text = (raw or "").strip()

    fence = _FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()

    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    ends = [i for i in (text.rfind("}"), text.rfind("]")) if i != -1]
    if starts and ends and min(starts) < max(ends):
        return text[min(starts) : max(ends) + 1]
    return text


def validate_json(schema, raw_json):
    """Extract the JSON from raw_json and validate it against schema.
    Raises pydantic.ValidationError if it doesn't match."""
    return schema.model_validate_json(extract_json(raw_json))


def format_errors(exc):
    """Turn a pydantic ValidationError into short, readable problems,
    e.g. 'fact: Field required'."""
    problems = []
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "response"
        problems.append(f"{where}: {err['msg']}")
    return problems


def run_validators(result, validators):
    """Run caller-supplied checks on a schema-valid result.

    Each validator takes the parsed object and returns None if it's fine,
    or a string describing what's wrong. Returns the list of problems.
    """
    problems = []
    for check in validators or []:
        problem = check(result)
        if problem:
            problems.append(problem)
    return problems


def build_feedback(raw_response, problems):
    """The correction message that goes into the next attempt's prompt:
    what the model said, and what was wrong with it."""
    echoed = (raw_response or "")[:_MAX_ECHO_CHARS]
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        f"Your previous response was:\n{echoed}\n\n"
        f"It had these problems:\n{listed}"
    )


async def validate_with_retry(schema, call_fn, max_attempts=3, validators=None):
    """
    Calls call_fn(feedback) to get raw text, then checks it: parse,
    schema, then custom validators. On any failure, calls call_fn again
    with feedback describing the problems. Gives up after max_attempts
    total tries (the first try counts) and raises ValidationRetryExceeded.
    """
    feedback = None
    raw = None
    problems = []

    for attempt in range(1, max_attempts + 1):
        raw = await call_fn(feedback)

        try:
            result = validate_json(schema, raw)
        except ValidationError as e:
            problems = format_errors(e)
        else:
            problems = run_validators(result, validators)
            if not problems:
                return result

        feedback = build_feedback(raw, problems)
        log.warning("response failed validation", attempt=attempt, errors=problems)

    raise ValidationRetryExceeded(
        f"response failed validation after {max_attempts} attempts",
        attempts=max_attempts,
        last_error="; ".join(problems),
        last_response=raw,
        errors=problems,
    )