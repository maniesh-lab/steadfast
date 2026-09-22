"""
Shared pytest fixtures. Anything defined here is automatically
available to every test file in this folder — no import needed.
"""
import pytest
from pydantic import BaseModel


class Fact(BaseModel):
    topic: str
    fact: str


@pytest.fixture
def fact_schema():
    """A small reusable schema for tests that need *some* pydantic
    model to validate against, without caring what it represents."""
    return Fact