"""
Example usage of the agent SDK.

Run with: python example.py
Requires: OPENAI_API_KEY environment variable
"""

import asyncio
import os
from agent_sdk import create_provider, run_agent, tool, logging_hook


# Define tools
@tool(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g., 'Tokyo', 'New York'"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit",
                "default": "celsius"
            }
        },
        "required": ["location"]
    }
)
async def get_weather(location: str, unit: str = "celsius") -> str:
    # Mock implementation
    return f"Weather in {location}: 22°{'C' if unit == 'celsius' else 'F'}, partly cloudy"


@tool(
    name="calculate",
    description="Perform a mathematical calculation",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate, e.g., '2 + 2', '10 * 5'"
            }
        },
        "required": ["expression"]
    }
)
async def calculate(expression: str) -> str:
    # Simple eval (in production, use a safe math parser)
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


async def main():
    # Create provider (uses OPENAI_API_KEY env var by default)
    provider = create_provider(model="gpt-4o")

    # Conversation state
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": "What's the weather in Tokyo? Also, what's 123 * 456?"}
    ]

    # Run agent with logging
    print("Running agent...\n")
    response = await run_agent(
        provider=provider,
        messages=messages,
        tools=[get_weather, calculate],
        hooks=[logging_hook]
    )

    print(f"\n--- Final Response ---\n{response}")

    # Show conversation history
    print(f"\n--- Conversation ({len(messages)} messages) ---")
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "tool":
            print(f"[tool] {content[:80]}...")
        elif role == "assistant" and msg.get("tool_calls"):
            tools = [tc["function"]["name"] for tc in msg["tool_calls"]]
            print(f"[assistant] Calling tools: {tools}")
        else:
            preview = content[:80] if content else "(no content)"
            print(f"[{role}] {preview}")


if __name__ == "__main__":
    asyncio.run(main())
