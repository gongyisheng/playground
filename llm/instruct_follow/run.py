from __future__ import annotations

import asyncio
from collections import Counter

from openai import AsyncOpenAI
from tqdm import tqdm


PROMPTS = [
    (
        "say hello, just the word hello",
        "hello",
    ),
    (
        "if user says 1, return A. if user says 2, return B. user says 1",
        "A",
    ),
]

RUNS_PER_PROMPT = 100


async def run_prompt(client: AsyncOpenAI, prompt: str, expected: str) -> dict:
    counts: Counter[str] = Counter()

    for _ in tqdm(range(RUNS_PER_PROMPT), desc="Runs", leave=False):
        response = await client.completions.create(
            model="gpt-5-mini", prompt=prompt
        )
        text = response.choices[0].text
        counts[text] += 1

    successes = counts.get(expected, 0)
    return {
        "prompt": prompt,
        "expected": expected,
        "successes": successes,
        "total": RUNS_PER_PROMPT,
        "accuracy": successes / RUNS_PER_PROMPT,
        "counts": counts,
    }


async def main() -> None:
    client = AsyncOpenAI()

    print("Instruct-following stability test (exact match)")
    print(f"Model: gpt-5-mini | Runs per prompt: {RUNS_PER_PROMPT}\n")

    for prompt, expected in PROMPTS:
        result = await run_prompt(client, prompt, expected)
        accuracy_pct = result["accuracy"] * 100
        print("Prompt:")
        print(prompt)
        print(f"Expected: {expected!r}")
        print(
            "Accuracy: "
            f"{result['successes']}/{result['total']} ({accuracy_pct:.1f}%)"
        )

        print("Output frequencies (desc):")
        for text, count in result["counts"].most_common():
            print(f"  {count:>3}  {text!r}")

        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
