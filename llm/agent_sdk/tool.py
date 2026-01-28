"""
Tool definition and execution.
"""

from typing import Callable, Awaitable
from .hooks import emit_hooks


def tool(
    name: str,
    description: str,
    parameters: dict,
    max_retries: int = 3
):
    """
    Decorator to define a tool.

    Args:
        name: Tool name (used in function calls)
        description: What the tool does (shown to LLM)
        parameters: JSON schema for parameters, following OpenAI format:
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        max_retries: Number of retry attempts on failure

    Returns:
        Tool definition dict

    Example:
        @tool(
            name="search_web",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        )
        async def search_web(query: str) -> str:
            return f"Results for: {query}"
    """
    def decorator(fn: Callable[..., Awaitable[str]]) -> dict:
        return {
            "name": name,
            "description": description,
            "parameters": parameters,
            "fn": fn,
            "max_retries": max_retries
        }
    return decorator


def to_openai_schema(tool_def: dict) -> dict:
    """Convert tool definition to OpenAI function schema."""
    return {
        "type": "function",
        "function": {
            "name": tool_def["name"],
            "description": tool_def["description"],
            "parameters": tool_def["parameters"]
        }
    }


async def execute_tool(
    tool_def: dict,
    args: dict,
    hooks: list | None = None
) -> dict:
    """
    Execute a tool with retry logic.

    Args:
        tool_def: Tool definition from @tool decorator
        args: Arguments to pass to the tool function
        hooks: List of hook functions for observability

    Returns:
        ToolResult dict with success, output, and error fields
    """
    hooks = hooks or []
    last_error = None

    for attempt in range(tool_def["max_retries"]):
        try:
            result = await tool_def["fn"](**args)
            return {"success": True, "output": result, "error": None}
        except Exception as e:
            last_error = str(e)
            if attempt < tool_def["max_retries"] - 1:
                await emit_hooks(hooks, "tool_retry", {
                    "tool": tool_def["name"],
                    "attempt": attempt + 1,
                    "error": last_error
                })

    return {"success": False, "output": None, "error": last_error}
