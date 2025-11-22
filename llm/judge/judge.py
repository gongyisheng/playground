from typing import Dict, List, Optional
import math
from functools import lru_cache
from pathlib import Path
import yaml
import os

from openai import AsyncOpenAI


class LLMJudge:
    """OpenAI models tokenize all numbers from 0-100 as single tokens, which is why we can get exactly
    one completion token with logprobs. Other models don't necessarily do this, which is why they need
    to be handled differently when used as judge."""

    PROVIDERS = {
        'openai': {
            'base_url': None,
            'api_key_env': 'OPENAI_API_KEY'
        },
        'openrouter': {
            'base_url': 'https://openrouter.ai/api/v1',
            'api_key_env': 'OPENROUTER_API_KEY'
        }
    }

    def __init__(self, model: str, prompt_template: str, provider: str = 'openai'):
        self.model = model
        self.prompt_template = prompt_template
        self.provider = provider

        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(self.PROVIDERS.keys())}")

        provider_config = self.PROVIDERS[provider]
        base_url = provider_config['base_url']
        api_key = os.getenv(provider_config['api_key_env'])

        if not api_key:
            raise ValueError(f"API key not found in environment variable: {provider_config['api_key_env']}")

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def judge(self, **kwargs):
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))
        
        return result
    
    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explcitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total
    
    async def __call__(self, **kwargs):
        return await self.judge(**kwargs)


async def test():
    model = "gpt-4.1-nano"
    prompt = "give me a number between 0 to 100, only number output"
    judge = LLMJudge(model, prompt)
    print(await judge.judge())

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())