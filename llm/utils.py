import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Any


def init_openai_client(*args, **kwargs) -> AsyncOpenAI:
    """
    Initialize and return an AsyncOpenAI client instance.

    Args:
        *args: Positional arguments to pass to AsyncOpenAI constructor
        **kwargs: Keyword arguments to pass to AsyncOpenAI constructor
                 (e.g., api_key, base_url, timeout, etc.)

    Returns:
        AsyncOpenAI client instance
    """
    return AsyncOpenAI(*args, **kwargs)


async def evoke_single_request(
    client: AsyncOpenAI,
    input,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Make a single async request to OpenAI API.

    Args:
        client: AsyncOpenAI client instance
        prompt: Input prompt text
        *args: Additional positional arguments to pass to completions.create()
        **kwargs: Additional keyword arguments to pass to completions.create()
                 (e.g., model, max_tokens, temperature, top_p, frequency_penalty,
                 presence_penalty, stop, logprobs, n, best_of, etc.)

    Returns:
        Dictionary containing the API response and metadata
    """
    try:
        response = await client.responses.create(
            input=input,
            *args,
            **kwargs
        )

        return response
    except Exception as e:
        return None


async def evoke_batch_requests(
    inputs,
    *args,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Make batch requests to OpenAI API using asyncio.

    This function sends multiple prompts concurrently to the OpenAI API,
    significantly improving throughput compared to sequential requests.

    Args:
        inputs: List of input messages to process
        *args: Additional positional arguments to pass to completions.create()
        **kwargs: Additional keyword arguments to pass to completions.create()
                 (e.g., model, max_tokens, temperature, top_p, frequency_penalty,
                 presence_penalty, stop, logprobs, n, best_of, etc.)

    Returns:
        List of result dictionaries, one per input
    """
    client = init_openai_client()

    tasks = [
        asyncio.create_task(
            evoke_single_request(
                client,
                input,
                *args,
                **kwargs
            )
        )
        for input in inputs
    ]

    results = await asyncio.gather(*tasks)

    return results

async def test():
    client = init_openai_client()
    prompt = "What's capital of France?"
    response = await evoke_single_request(client, prompt, model="gpt-5-mini")
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())