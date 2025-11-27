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
    messages,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Make a single async request to OpenAI API.

    Args:
        client: AsyncOpenAI client instance
        message: Input message context
        *args: Additional positional arguments to pass to completions.create()
        **kwargs: Additional keyword arguments to pass to completions.create()
                 (e.g., model, max_tokens, temperature, top_p, frequency_penalty,
                 presence_penalty, stop, logprobs, n, best_of, etc.)

    Returns:
        Dictionary containing the API response and metadata
    """
    try:
        response = await client.chat.completions.create(
            messages=messages,
            *args,
            **kwargs
        )
        return response
    except Exception as e:
        print(e)
        return None


async def evoke_batch_requests(
    client,
    messages_list,
    *args,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Make batch requests to OpenAI API using asyncio.

    This function sends multiple prompts concurrently to the OpenAI API,
    significantly improving throughput compared to sequential requests.

    Args:
        messages_list: List of input messages to process
        *args: Additional positional arguments to pass to completions.create()
        **kwargs: Additional keyword arguments to pass to completions.create()
                 (e.g., model, max_tokens, temperature, top_p, frequency_penalty,
                 presence_penalty, stop, logprobs, n, best_of, etc.)

    Returns:
        List of result dictionaries, one per input
    """

    tasks = [
        asyncio.create_task(
            evoke_single_request(
                client,
                messages,
                *args,
                **kwargs
            )
        )
        for messages in messages_list
    ]

    results = await asyncio.gather(*tasks)

    return results

async def test():
    client = init_openai_client()
    messages = [{"role": "user", "content": "What's capital of France?"}]
    response = await evoke_single_request(client, messages, model="gpt-5-mini")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())