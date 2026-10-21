"""
Response validation for Steadfast.

Turns raw LLM text into a validated Pydantic object. These kinds of
failure are caught, and all of them are fed back to the model so it can
fix its own mistake:

1. Not parseable   - no valid JSON in the reply
2. Wrong shape     - valid JSON, but it doesn't match the schema
3. Failed a check  - matches the schema, but a custom validator rejected it
4. Broken tool call - the model asked for a tool with arguments that
                      aren't valid JSON

The client (client.py) drives the loop with check_response() and
feedback_messages(). validate_with_retry() is the older, prompt-string
version of the same loop, kept for anyone using it directly.
"""

import inspect
import re

from pydantic import ValidationError

from .exceptions import ValidationRetryExceeded
from .logging_utils import get_logger

log = get_logger()

# Matches ```json ... ``` (or plain ``` ... ```) code fences.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Don't paste an enormous bad reply back into the next prompt.
_MAX_ECHO_CHARS = 2000

_CUT_OFF_PROBLEM = "your reply was cut off because it hit the token limit - give a shorter answer"


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
    (Sync only; the client uses run_validators_async, which also accepts
    async validators.)
    """
    problems = []
    for check in validators or []:
        problem = check(result)
        if problem:
            problems.append(problem)
    return problems


async def run_validators_async(result, validators):
    """Like run_validators, but a validator may also be an async function."""
    problems = []
    for check in validators or []:
        problem = check(result)
        if inspect.isawaitable(problem):
            problem = await problem
        if problem:
            problems.append(problem)
    return problems


def build_feedback(raw_response, problems):
    """The correction message: what the model said, and what was wrong
    with it. Used by validate_with_retry (single-prompt style)."""
    echoed = (raw_response or "")[:_MAX_ECHO_CHARS]
    listed = "\n".join(f"- {p}" for p in problems)
    return f"Your previous response was:\n{echoed}\n\nIt had these problems:\n{listed}"


async def check_response(response, schema=None, validators=None):
    """Check one LLMResponse. Returns (parsed, problems).

    - Broken tool-call arguments are always a problem.
    - If the model called tools, the schema is skipped: a tool call is a
      valid answer, and the schema only describes final text answers.
    - Otherwise, with a schema: parse, validate, run validators.
    """
    broken = [call for call in response.tool_calls if call.arguments is None]
    if broken:
        return None, [
            f"tool call '{call.name}' had arguments that are not valid JSON: "
            f"{call.raw_arguments[:500]}"
            for call in broken
        ]

    if schema is None or response.tool_calls:
        return None, []

    try:
        parsed = validate_json(schema, response.text)
    except ValidationError as e:
        problems = format_errors(e)
    else:
        problems = await run_validators_async(parsed, validators)
        if not problems:
            return parsed, []

    if response.finish_reason == "length":
        problems.append(_CUT_OFF_PROBLEM)
    return None, problems


def feedback_messages(response, problems):
    """The messages to add before the next attempt, so the model sees
    its own mistake and the list of problems."""
    listed = "\n".join(f"- {p}" for p in problems)

    if any(call.arguments is None for call in response.tool_calls):
        # Don't echo the broken tool call itself: providers require every
        # tool call to be answered with a result, which we can't do here.
        return [
            {
                "role": "user",
                "content": (
                    "Your previous reply tried to call a tool, but it had these problems:\n"
                    f"{listed}\n\nPlease call the tool again with valid JSON arguments."
                ),
            }
        ]

    return [
        {"role": "assistant", "content": (response.text or "")[:_MAX_ECHO_CHARS]},
        {
            "role": "user",
            "content": (
                f"Your previous response had these problems:\n{listed}\n\n"
                "Please fix these problems and respond again with only the corrected JSON."
            ),
        },
    ]


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
            problems = await run_validators_async(result, validators)
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
