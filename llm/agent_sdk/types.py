"""
Type definitions for the agent SDK.
"""

from typing import TypedDict, Callable, Awaitable, Any


class Message(TypedDict, total=False):
    """Chat message format following OpenAI schema."""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None
    tool_calls: list
    tool_call_id: str


class ToolDef(TypedDict):
    """
    Tool definition.

    Schema follows OpenAI's function calling format:
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        },
        "fn": <async function>,
        "max_retries": 3
    }
    """
    name: str
    description: str
    parameters: dict  # JSON schema
    fn: Callable[..., Awaitable[str]]
    max_retries: int


class ToolResult(TypedDict):
    """Result of a tool execution."""
    success: bool
    output: str | None
    error: str | None


class Provider(TypedDict):
    """Provider configuration."""
    client: Any  # AsyncOpenAI
    model: str


# Hook function signature: (event: str, data: dict) -> None | Awaitable[None]
HookFn = Callable[[str, dict], None | Awaitable[None]]
