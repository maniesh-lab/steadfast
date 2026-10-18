"""
Helpers for describing your tools to the model.

A tool is a function the model may ask you to run. You describe it with
a name, a sentence about what it does, and a JSON schema for its inputs.
"""

from __future__ import annotations

from typing import Any


def function_tool(name: str, description: str, parameters: Any = None) -> dict[str, Any]:
    """Build a tool definition in OpenAI's format (Steadfast translates
    it for other providers).

    parameters: a JSON-schema dict, or a Pydantic model class whose
        fields are the tool's inputs. None means "no inputs".

    Example:
        class SearchArgs(BaseModel):
            query: str
            limit: int = 5

        search = function_tool("search_docs", "Search the policy documents.", SearchArgs)
    """
    if parameters is None:
        schema: dict[str, Any] = {"type": "object", "properties": {}}
    elif isinstance(parameters, dict):
        schema = parameters
    elif hasattr(parameters, "model_json_schema"):
        schema = parameters.model_json_schema()
    else:
        raise TypeError("parameters must be a JSON-schema dict or a Pydantic model class")

    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }
