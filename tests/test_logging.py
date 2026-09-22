"""
Tests for logging_utils.py — proves correlation IDs actually get bound
during a block and cleaned up after it, not just that the code runs.
"""
import structlog

from steadfast.logging_utils import correlation_context


def test_correlation_context_yields_a_string_id():
    with correlation_context() as cid:
        assert isinstance(cid, str)
        assert len(cid) > 0


def test_correlation_context_uses_provided_id():
    with correlation_context(correlation_id="fixed-id-123") as cid:
        assert cid == "fixed-id-123"


def test_different_blocks_get_different_ids():
    with correlation_context() as first:
        pass
    with correlation_context() as second:
        pass
    assert first != second


def test_correlation_id_is_actually_bound_during_block():
    with correlation_context() as cid:
        bound = structlog.contextvars.get_contextvars()
        assert bound.get("correlation_id") == cid


def test_correlation_id_is_unbound_after_block():
    with correlation_context():
        pass
    bound = structlog.contextvars.get_contextvars()
    assert "correlation_id" not in bound