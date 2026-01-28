"""
Provider wrapper for OpenAI-compatible APIs.

Works with: OpenAI, Azure OpenAI, Groq, Together, Ollama, vLLM, etc.
"""

from openai import AsyncOpenAI
from typing import Any


def create_provider(
    model: str,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    """
    Create a provider configuration.

    Args:
        model: Model identifier (e.g., "gpt-4o", "llama-3.1-70b")
        base_url: API base URL. None for OpenAI, or custom endpoint for others:
                  - Groq: "https://api.groq.com/openai/v1"
                  - Together: "https://api.together.xyz/v1"
                  - Ollama: "http://localhost:11434/v1"
        api_key: API key. None will use OPENAI_API_KEY env var.

    Returns:
        Provider dict with client and model.
    """
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return {
        "client": client,
        "model": model
    }


async def chat(
    provider: dict,
    messages: list,
    tools: list[dict] | None = None
) -> Any:
    """
    Call the LLM.

    Args:
        provider: From create_provider()
        messages: List of chat messages
        tools: OpenAI function schemas (optional)

    Returns:
        ChatCompletion response
    """
    kwargs = {
        "model": provider["model"],
        "messages": messages
    }
    if tools:
        kwargs["tools"] = tools

    return await provider["client"].chat.completions.create(**kwargs)
