"""
Response validation for Steadfast.

Validates raw LLM output against a Pydantic schema. If it doesn't match,
the validation error is fed back to the model as a correction hint and
the call is retried — usually cheaper and more reliable than a single
hard-coded parser giving up on the first bad response.
"""
from typing import Awaitable, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import ValidationRetryExceeded
from .logging_utils import get_logger

log = get_logger()

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def validate_json(schema: type[SchemaT], raw_json: str) -> SchemaT:
    """
    Parse and validate raw_json against schema.

    Raises pydantic.ValidationError directly if it doesn't match —
    validate_with_retry() is what turns that into a retry loop.
    """
    return schema.model_validate_json(raw_json)


async def validate_with_retry(
    schema: type[SchemaT],
    call_fn: Callable[[str | None], Awaitable[str]],
    max_attempts: int = 3,
) -> SchemaT:
    """
    Calls call_fn() to get raw text, validates it against schema. On
    failure, calls call_fn(repair_hint) again with the validation error
    included, so the model gets a chance to correct itself.

    call_fn signature: async def call_fn(repair_hint: str | None) -> str
        - first call: repair_hint is None
        - retry calls: repair_hint describes what went wrong last time

    Raises ValidationRetryExceeded if every attempt fails.
    """
    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        raw = await call_fn(last_error)
        try:
            return validate_json(schema, raw)
        except ValidationError as e:
            last_error = str(e)
            log.warning(
                "response failed validation",
                attempt=attempt,
                schema=schema.__name__,
                error=last_error,
            )

    raise ValidationRetryExceeded(
        f"response failed validation after {max_attempts} attempts",
        attempts=max_attempts,
        last_error=last_error,
    )