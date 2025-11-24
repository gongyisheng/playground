from typing import Dict, List, Optional, Any, Literal
import math
import os
import asyncio
import traceback
import re
from pathlib import Path
from omegaconf import OmegaConf

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

    def __init__(
            self,
            prompt_template: str,
            provider: str,
            model: str,
            capabilities_file: Optional[str] = "model_capabilities.yaml"
        ):
        self.model = model
        self.prompt_template = prompt_template
        self.provider = provider
        self._logged_unsupported_params = False  # Track if we've already logged unsupported params

        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(self.PROVIDERS.keys())}")

        provider_config = self.PROVIDERS[provider]
        base_url = provider_config['base_url']
        api_key = os.getenv(provider_config['api_key_env'])

        if not api_key:
            raise ValueError(f"API key not found in environment variable: {provider_config['api_key_env']}")

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        # Load model capabilities
        self.capabilities = self._load_capabilities(capabilities_file)

    def _load_capabilities(self, capabilities_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model capabilities from YAML file.

        Args:
            capabilities_file: Path to capabilities YAML file. If None, looks for
                             model_capabilities.yaml in the same directory as this file.

        Returns:
            Dictionary of model capabilities
        """
        if capabilities_file is None:
            # Default to model_capabilities.yaml in the same directory
            capabilities_file = Path(__file__).parent / 'model_capabilities.yaml'
        else:
            capabilities_file = Path(capabilities_file)

        if not capabilities_file.exists():
            raise FileNotFoundError(f"Model capabilities file not found: {capabilities_file}")

        # Load with OmegaConf to support variable interpolation
        config = OmegaConf.load(capabilities_file)
        # Convert to dict, resolving all interpolations
        data = OmegaConf.to_container(config, resolve=True)

        return data.get('model_capabilities', {})

    def _get_model_capabilities(self) -> Dict[str, bool]:
        """
        Get capabilities for the current model.
        Uses prefix matching - finds the longest matching prefix from available models.

        Returns:
            Dictionary mapping capability names to boolean support values
        """
        provider_caps = self.capabilities.get(self.provider, {})

        # Try prefix matching - find the longest matching prefix
        # Sort by length descending to prioritize longer (more specific) prefixes
        matching_prefix = None
        for model_prefix in sorted(provider_caps.keys(), key=len, reverse=True):
            if model_prefix != 'default' and self.model.startswith(model_prefix):
                matching_prefix = model_prefix
                break

        if matching_prefix:
            return provider_caps[matching_prefix]

        # Fall back to default for this provider
        if 'default' in provider_caps:
            return provider_caps['default']

        # Ultimate fallback - assume no special capabilities
        return {
            'logprobs': False,
            'temperature': True,
            'top_logprobs': False,
            'seed': False,
            'max_tokens': True
        }

    def _filter_params_by_capability(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter API parameters based on model capabilities.
        Logs which parameters are removed.

        Args:
            params: Original API parameters

        Returns:
            Filtered parameters with unsupported ones removed
        """
        capabilities = self._get_model_capabilities()
        filtered = params.copy()
        removed = []

        # Check max_tokens
        if not capabilities.get('max_tokens', True):
            if filtered.pop('max_tokens', None) is not None:
                removed.append('max_tokens')

        # Check logprobs
        if not capabilities.get('logprobs', False):
            if filtered.pop('logprobs', None) is not None:
                removed.append('logprobs')
            if filtered.pop('top_logprobs', None) is not None:
                removed.append('top_logprobs')

        # Check temperature
        if not capabilities.get('temperature', True):
            if filtered.pop('temperature', None) is not None:
                removed.append('temperature')

        # Check seed
        if not capabilities.get('seed', False):
            if filtered.pop('seed', None) is not None:
                removed.append('seed')

        # Log removed parameters (only once per instance)
        if removed and not self._logged_unsupported_params:
            print(f"Model {self.model} doesn't support: {', '.join(removed)}")
            self._logged_unsupported_params = True

        return filtered

    async def _call_openai_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1,
        temperature: float = 0.0,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Centralized OpenAI API call function.

        Args:
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            logprobs: Whether to return logprobs
            top_logprobs: Number of top logprobs to return
            seed: Random seed for reproducibility

        Returns:
            Completion object from OpenAI API
        """
        params = {
            'model': self.model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
        }

        if logprobs:
            params['logprobs'] = True
            if top_logprobs is not None:
                params['top_logprobs'] = top_logprobs

        if seed is not None:
            params['seed'] = seed

        # Filter parameters based on model capabilities
        params = self._filter_params_by_capability(params)

        return await self.client.chat.completions.create(**params)

    async def logprob_weighted_judge(
        self,
        min_score: int,
        max_score: int,
        temperature: float = 0.0,
        max_tokens: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Judge using logprobs with weighted scoring.

        Args:
            min_score: lower bound of score
            max_score: upper bound of score
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Template variables for the prompt

        Returns:
            dict with keys:
                - 'score': weighted score based on logprobs
                - 'distribution': dict mapping each score to its probability
                - 'method': 'logprob_weighted'
        """

        # Validate score range for logprob method
        if min_score < 0 or max_score > 9:
            print("logprob_weighted_judge works best with scores 0-9. This may not work correctly as numbers outside 0-9 may not be single tokens.")

        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]

        # Call API with logprobs enabled
        completion = await self._call_openai_api(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )

        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return None

        # Convert logprobs to probabilities
        distribution = {}
        for el in logprobs:
            distribution[el.token] = float(math.exp(el.logprob))

        # Calculate weighted score
        score = self._aggregate_score(distribution, min_score, max_score)

        return {
            'score': score
        }

    async def monte_carlo_judge(
        self,
        min_score: int,
        max_score: int,
        num_rounds: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 100,
        score_pattern: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Judge using multiple completion rounds and averaging.

        Args:
            min_score: lower bound of score
            max_score: upper bound of score
            num_rounds: Number of rounds to run
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            score_pattern: Optional regex pattern to extract score from completion.
            **kwargs: Template variables for the prompt

        Returns:
            dict with key 'score': mean score returned by monte carlo judge
        """
        messages = [dict(role='user', content=self.prompt_template.format(**kwargs))]

        # Run multiple rounds concurrently
        tasks = [
            self._call_openai_api(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=None  # No seed for sampling diversity
            )
            for _ in range(num_rounds)
        ]

        completions = await asyncio.gather(*tasks)

        # Parse scores from completions and build distribution
        scores = []
        for completion in completions:
            try:
                content = completion.choices[0].message.content
                match = re.search(score_pattern, content)
                if match:
                    score = match.group(1)
                    scores.append(score)
            except Exception:
                continue

        if not scores:
            return None

        # Build distribution: count occurrences and convert to probabilities
        distribution = {}
        for score in scores:
            distribution[score] = distribution.get(score, 0) + 1

        # Normalize to probabilities
        total_count = len(scores)
        for key in distribution:
            distribution[key] = distribution[key] / total_count

        # Use aggregate_score to calculate weighted score
        score = self._aggregate_score(distribution, min_score, max_score)

        return {
            'score': score
        }

    def _aggregate_score(self, distribution: dict, min_score: int, max_score: int) -> Optional[float]:
        """
        Aggregate weighted score from probability distribution.

        Args:
            distribution: Dict mapping tokens to probabilities
            min_score: lower bound of score
            max_score: upper bound of score

        Returns:
            Weighted score or None if aggregation fails
        """

        # NOTE: we don't check for refusals explicitly. Instead we assume that
        # if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in distribution.items():
            try:
                score = float(key)
            except ValueError:
                continue
            if score < min_score or score > max_score:
                continue
            sum_ += score * val
            total += val

        if total < 0.25:
            # Failed to aggregate because total weight on valid scores is less than 0.25.
            return None
        return sum_ / total
    
    async def judge(
        self,
        min_score: int,
        max_score: int,
        method: Literal['logprob_weighted', 'monte_carlo'] = 'logprob_weighted',
        temperature: float = 0.7,
        max_tokens: int = 1,
        monte_carlo_num_rounds: int = 10,
        monte_carlo_score_pattern: Optional[str] = None,
        **kwargs
    ):
        """
        Main entry point for judging.

        Args:
            min_score: lower bound of score
            max_score: upper bound of score
            method: Judge method to use ('logprob_weighted' or 'monte_carlo')
            num_rounds: Number of rounds for monte_carlo method
            temperature: Sampling temperature (for both methods)
            max_tokens: Maximum tokens to generate (for both methods)
            score_pattern: Optional regex pattern to extract score for monte_carlo method
            **kwargs: Template variables for the prompt

        Returns:
            dict with score and method-specific metrics
        """
        # Validate method compatibility with model capabilities
        if method == 'logprob_weighted':
            capabilities = self._get_model_capabilities()
            if not capabilities.get('logprobs', False):
                raise ValueError(
                    f"Model '{self.model}' doesn't support logprobs, which is required for method='logprob_weighted'. "
                    f"Please use method='monte_carlo' instead."
                )

            return await self.logprob_weighted_judge(
                min_score,
                max_score,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        elif method == 'monte_carlo':
            return await self.monte_carlo_judge(
                min_score,
                max_score,
                temperature=temperature,
                max_tokens=max_tokens,
                num_rounds=monte_carlo_num_rounds,
                score_pattern=monte_carlo_score_pattern,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported method: {method}. Supported methods: ['logprob_weighted', 'monte_carlo']")
    
    async def judge_with_retry(self, retry: int = 3, **kwargs) -> Optional[Dict[str, Any]]:
        """Judge with retry logic. Same signature as judge() but with retry parameter."""
        for attempt in range(retry + 1):
            try:
                result = await self.judge(**kwargs)
                return result
            except Exception as e:
                if attempt == retry:
                    # All retries exhausted
                    print(f"\nFailed to judge after {retry + 1} attempts: {e}")
                    traceback.print_exc()
                    return None

    async def judge_batch(self, tasks: List[Dict[str, Any]], retry: int = 3, judge_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Judge a batch of items concurrently.

        Args:
            tasks: List of dicts containing template kwargs and original data
            retry: Number of retry attempts for failed requests
            judge_kwargs: Judge configuration (method, min_score, max_score, etc.) to apply to all tasks

        Returns:
            List of dicts with original data + judge result fields (score)
        """
        if judge_kwargs is None:
            judge_kwargs = {}

        # Execute all tasks concurrently, merging judge_kwargs with each task
        results = await asyncio.gather(*[
            asyncio.create_task(self.judge_with_retry(retry=retry, **{**task, **judge_kwargs}))
            for task in tasks
        ])

        # Combine results
        combined_results = []
        for idx, result in enumerate(results):
            result_dict = tasks[idx].copy()
            if result is not None:
                result_dict.update(result)
            else:
                result_dict['score'] = None
            combined_results.append(result_dict)
        return combined_results


async def test():
    provider = "openai"
    model = "gpt-5-mini"
    prompt = "give me a number between 0 to 9, only number output"
    judge = LLMJudge(prompt, provider=provider, model=model)
    # print(await judge.judge(0, 9, "logprob_weighted"))
    print(await judge.judge(0, 9, "monte_carlo", monte_carlo_score_pattern=r'^(\d)$'))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())