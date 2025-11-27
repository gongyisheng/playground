import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional


async def make_single_request(
    client: AsyncOpenAI,
    engine: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.5,
    frequency_penalty: float = 0,
    presence_penalty: float = 2,
    stop_sequences: Optional[List[str]] = None,
    logprobs: Optional[int] = None,
    n: int = 1,
    best_of: int = 1,
) -> Dict[str, Any]:
    """
    Make a single async request to OpenAI API.

    Args:
        client: AsyncOpenAI client instance
        engine: Model engine to use (e.g., 'gpt-3.5-turbo-instruct', 'davinci-002')
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter
        frequency_penalty: Penalty for token frequency
        presence_penalty: Penalty for token presence
        stop_sequences: Sequences where generation should stop
        logprobs: Include log probabilities
        n: Number of completions to generate
        best_of: Generate best_of completions and return the best n

    Returns:
        Dictionary containing the API response and metadata
    """
    try:
        response = await client.completions.create(
            model=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop_sequences,
            logprobs=logprobs,
            n=n,
            best_of=best_of,
        )

        return {
            "response": response.model_dump(),
            "prompt": prompt,
            "success": True,
        }
    except Exception as e:
        return {
            "response": None,
            "prompt": prompt,
            "success": False,
            "error": str(e),
        }


async def make_requests_async(
    engine: str,
    prompts: List[str],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.5,
    frequency_penalty: float = 0,
    presence_penalty: float = 2,
    stop_sequences: Optional[List[str]] = None,
    logprobs: Optional[int] = None,
    n: int = 1,
    best_of: int = 1,
) -> List[Dict[str, Any]]:
    """
    Make batch requests to OpenAI API using asyncio.

    This function sends multiple prompts concurrently to the OpenAI API,
    significantly improving throughput compared to sequential requests.

    Args:
        engine: Model engine to use (e.g., 'gpt-3.5-turbo-instruct', 'davinci-002')
        prompts: List of prompt strings to process
        max_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter
        frequency_penalty: Penalty for token frequency
        presence_penalty: Penalty for token presence
        stop_sequences: Sequences where generation should stop
        logprobs: Include log probabilities
        n: Number of completions to generate per prompt
        best_of: Generate best_of completions and return the best n

    Returns:
        List of result dictionaries, one per prompt
    """
    client = AsyncOpenAI()

    tasks = [
        make_single_request(
            client=client,
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            logprobs=logprobs,
            n=n,
            best_of=best_of,
        )
        for prompt in prompts
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions that occurred
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "response": None,
                "prompt": prompts[i],
                "success": False,
                "error": str(result),
            })
        else:
            processed_results.append(result)

    return processed_results


def make_requests(
    engine: str,
    prompts: List[str],
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.5,
    frequency_penalty: float = 0,
    presence_penalty: float = 2,
    stop_sequences: Optional[List[str]] = None,
    logprobs: Optional[int] = None,
    n: int = 1,
    best_of: int = 1,
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for make_requests_async.

    This function provides a synchronous interface that can be called
    from non-async code, while using asyncio internally for batch requests.

    Args:
        Same as make_requests_async

    Returns:
        List of result dictionaries, one per prompt
    """
    return asyncio.run(
        make_requests_async(
            engine=engine,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            logprobs=logprobs,
            n=n,
            best_of=best_of,
        )
    )
