"""
Hook system for observability and logging.

Hook events:
- "pre_llm": before LLM call, data: {"messages": list}
- "post_llm": after LLM response, data: {"response": message}
- "pre_tool": before tool execution, data: {"tool": name, "args": dict}
- "post_tool": after tool execution, data: {"tool": name, "result": str}
- "tool_retry": on retry attempt, data: {"tool": name, "attempt": int, "error": str}
- "tool_error": after all retries failed, data: {"tool": name, "error": str}
"""

import asyncio
from typing import Callable


async def emit_hooks(hooks: list[Callable], event: str, data: dict):
    """Fire all registered hooks for an event."""
    for hook in hooks:
        if asyncio.iscoroutinefunction(hook):
            await hook(event, data)
        else:
            hook(event, data)


def logging_hook(event: str, data: dict):
    """Simple logging hook for debugging."""
    if event == "pre_llm":
        msg_count = len(data.get("messages", []))
        print(f"[pre_llm] Sending {msg_count} messages to LLM")
    elif event == "post_llm":
        response = data.get("response")
        has_tools = bool(response.tool_calls) if response else False
        print(f"[post_llm] Received response, has_tool_calls={has_tools}")
    elif event == "pre_tool":
        print(f"[pre_tool] Calling {data.get('tool')} with {data.get('args')}")
    elif event == "post_tool":
        result = data.get("result", "")
        preview = result[:100] + "..." if len(result) > 100 else result
        print(f"[post_tool] {data.get('tool')} returned: {preview}")
    elif event == "tool_retry":
        print(f"[tool_retry] {data.get('tool')} attempt {data.get('attempt')}: {data.get('error')}")
    elif event == "tool_error":
        print(f"[tool_error] {data.get('tool')} failed: {data.get('error')}")
