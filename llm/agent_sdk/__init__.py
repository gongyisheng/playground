"""
Minimal Agent SDK

A functional, minimal SDK for building LLM agents with tool use.

Example:
    from agent_sdk import create_provider, run_agent, tool
    from agent_sdk.hooks import logging_hook

    @tool(
        name="greet",
        description="Greet someone",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet"}
            },
            "required": ["name"]
        }
    )
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    provider = create_provider(model="gpt-4o")

    messages = [
        {"role": "user", "content": "Please greet Alice"}
    ]

    response = await run_agent(
        provider=provider,
        messages=messages,
        tools=[greet],
        hooks=[logging_hook]
    )
"""

from .agent import run_agent
from .provider import create_provider, chat
from .tool import tool, to_openai_schema, execute_tool
from .hooks import emit_hooks, logging_hook
from .types import Message, ToolDef, ToolResult, Provider, HookFn

__all__ = [
    # Core
    "run_agent",
    "create_provider",
    "chat",
    # Tools
    "tool",
    "to_openai_schema",
    "execute_tool",
    # Hooks
    "emit_hooks",
    "logging_hook",
    # Types
    "Message",
    "ToolDef",
    "ToolResult",
    "Provider",
    "HookFn",
]
