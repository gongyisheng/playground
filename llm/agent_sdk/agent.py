"""
Core agent loop.
"""

import json
import asyncio
from .provider import chat
from .tool import to_openai_schema, execute_tool
from .hooks import emit_hooks


async def run_agent(
    provider: dict,
    messages: list,
    tools: list[dict] | None = None,
    hooks: list | None = None,
    max_iterations: int = 20
) -> str:
    """
    Run the agent loop.

    The agent will:
    1. Send messages to the LLM
    2. If LLM requests tool calls, execute them in parallel
    3. Append results and repeat until LLM responds without tool calls

    Args:
        provider: From create_provider()
        messages: Conversation history (mutated in place with new messages)
        tools: List of tool definitions from @tool decorator
        hooks: List of hook functions for observability
        max_iterations: Safety limit to prevent infinite loops

    Returns:
        Final assistant response content

    Raises:
        RuntimeError: If max_iterations exceeded

    Example:
        provider = create_provider(model="gpt-4o", api_key="sk-...")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ]

        response = await run_agent(
            provider=provider,
            messages=messages,
            tools=[search_web, get_weather],
            hooks=[logging_hook]
        )
    """
    tools = tools or []
    hooks = hooks or []
    tool_map = {t["name"]: t for t in tools}
    openai_tools = [to_openai_schema(t) for t in tools] if tools else None

    for _ in range(max_iterations):
        # Pre-LLM hook
        await emit_hooks(hooks, "pre_llm", {"messages": messages})

        # Call LLM
        response = await chat(provider, messages, openai_tools)
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        # Post-LLM hook
        await emit_hooks(hooks, "post_llm", {"response": msg})

        # No tool calls - we're done
        if not msg.tool_calls:
            return msg.content

        # Execute tools in parallel
        tool_results = await _execute_tools_parallel(
            msg.tool_calls, tool_map, hooks
        )
        messages.extend(tool_results)

    raise RuntimeError(f"Agent exceeded max iterations ({max_iterations})")


async def _execute_tools_parallel(
    tool_calls,
    tool_map: dict,
    hooks: list
) -> list:
    """Execute multiple tool calls in parallel."""

    async def execute_one(tc) -> dict:
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        tool_def = tool_map.get(name)

        if not tool_def:
            return {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"Error: Unknown tool '{name}'"
            }

        # Pre-tool hook
        await emit_hooks(hooks, "pre_tool", {"tool": name, "args": args})

        # Execute with retry
        result = await execute_tool(tool_def, args, hooks)

        if result["success"]:
            await emit_hooks(hooks, "post_tool", {
                "tool": name,
                "result": result["output"]
            })
            content = result["output"]
        else:
            await emit_hooks(hooks, "tool_error", {
                "tool": name,
                "error": result["error"]
            })
            content = f"Tool failed after retries: {result['error']}"

        return {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": content
        }

    return await asyncio.gather(*[execute_one(tc) for tc in tool_calls])
