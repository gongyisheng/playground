import asyncio
import re
from typing import List

from openai import AsyncOpenAI

TEST_CASE = [
    "How many r are there in \'strawberry\'?",
    "Which is bigger, 9.9 or 9.11?",
]

async def a_generate(client, model_name, prompt: str):
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=4096
    )

    result = response.choices[0].message.content
    print(f"Question: {prompt}")
    print(f"Answer: {result}")
    return result

async def test():
    base_url = "http://localhost:11434/v1/"
    model_name = "qwen3:30b"
    client = AsyncOpenAI(base_url=base_url)
    tasks = []
    for case in TEST_CASE:
        task = asyncio.create_task(a_generate(client, model_name, case))
        tasks.append(task)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(test())