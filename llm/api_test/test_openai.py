#!/usr/bin/env python3
"""Simple CLI wrapper for calling GPT-5 Mini with reasoning enabled.

Usage:
    python openai.py "Explain quantum entanglement in simple terms"
    python openai.py --effort medium "Draft a product update"
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from openai import OpenAI


def require_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Export your OpenAI key before running this script."
    )


def create_client() -> OpenAI:
    require_api_key()
    return OpenAI()


def request_gpt5_mini(prompt: str, effort: str) -> Dict[str, Any]:
    client = create_client()
    return client.responses.create(
        model="gpt-5-mini",
        reasoning={
            "effort": effort, # minimal, low, medium, and high
            "summary": "detailed" # auto, concise, or detailed
        },
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            }
        ],
    )

def run(prompt: str, effort: str) -> None:
    response = request_gpt5_mini(prompt, effort)
    response_dict = response.model_dump()
    print(response_dict)

if __name__ == "__main__":
    prompt = "Which one is bigger, 9.9 or 9.11"
    effort = "high"
    run(prompt, effort)
