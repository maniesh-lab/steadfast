"""
Response validation for Steadfast.

Validates raw LLM output against a Pydantic schema. If it doesn't match,
the validation error is fed back to the model as a correction hint and
the call is retried.
"""

from pydantic import ValidationError
from .exceptions import ValidationRetryExceeded
from .logging_utils import get_logger

log = get_logger()


def validate_json(schema, raw_json):
    """Parse and validate raw_json against schema. Raises
    pydantic.ValidationError if it doesn't match."""
    return schema.model_validate_json(raw_json)


async def validate_with_retry(schema, call_fn, max_attempts=3):
    """
    Calls call_fn() to get raw text, validates it against schema.
    On failure, calls call_fn again, passing along what went wrong,
    so the model can try to fix it. Gives up after max_attempts.
    """
    last_error = None

    for attempt in range(1, max_attempts + 1):
        raw = await call_fn(last_error)
        try:
            return validate_json(schema, raw)
        except ValidationError as e:
            last_error = str(e)
            log.warning("response failed validation", attempt=attempt, error=last_error)

    raise ValidationRetryExceeded(
        f"response failed validation after {max_attempts} attempts",
        attempts=max_attempts,
        last_error=last_error,
    )