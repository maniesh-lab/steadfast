"""
Steadfast: a reliability layer for LLM API calls.

    from steadfast import SteadfastClient
    from steadfast.providers.groq_provider import GroqProvider
"""

from .client import SteadfastClient
from .exceptions import (
    AllProvidersFailed,
    NonRetryableProviderError,
    ProviderError,
    RetryableProviderError,
    SteadfastError,
    ValidationRetryExceeded,
)
from .logging_utils import configure_logging, correlation_context
from .tools import function_tool
from .types import LLMResponse, ToolCall, Usage, tool_result

__version__ = "0.3.0"

__all__ = [
    "AllProvidersFailed",
    "LLMResponse",
    "NonRetryableProviderError",
    "ProviderError",
    "RetryableProviderError",
    "SteadfastClient",
    "SteadfastError",
    "ToolCall",
    "Usage",
    "ValidationRetryExceeded",
    "configure_logging",
    "correlation_context",
    "function_tool",
    "tool_result",
]
