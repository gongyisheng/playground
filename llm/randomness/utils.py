import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

from openai import AsyncOpenAI

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
        client: openai client
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

async def test_openai():
    client = init_openai_client()
    messages = [{"role": "user", "content": "What's capital of France?"}]
    response = await evoke_single_request(client, messages, model="gpt-5-mini")
    print(response.choices[0].message.content)


class DataWriter:
    """
    Fault-tolerant incremental data writer for synthetic data generation.

    Features:
    - Automatic directory creation
    - Incremental writing (crash recovery)
    - Metadata tracking
    - Support for JSONL format
    - Resume capability from existing files
    - Automatic file handle management

    Usage:
        writer = DataWriter("outputs/data.jsonl")
        writer.write({"text": "example", "score": 0.95})
        writer.write_batch([{"text": "ex1"}, {"text": "ex2"}])
        writer.close()

        # Or use as context manager
        with DataWriter("outputs/data.jsonl") as writer:
            writer.write({"text": "example", "metadata": "value"})
    """

    def __init__(
        self,
        output_path: str,
        mode: str,
        auto_flush: bool = True,
        create_dirs: bool = True
    ):
        """
        Initialize DataWriter.

        Args:
            output_path: Path to output file (supports .jsonl, .json, .txt)
            mode: File open mode ('w' for write, 'a' for append)
            auto_flush: If True, flush after each write for crash recovery
            create_dirs: If True, automatically create parent directories
        """
        self.output_path = Path(output_path)
        self.mode = mode
        self.auto_flush = auto_flush
        self.create_dirs = create_dirs
        self.file_handle = None

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def open(self):
        """Open file handle."""
        if self.file_handle is None:
            # Create parent directories if needed
            if self.create_dirs:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)

            self.file_handle = open(self.output_path, self.mode)

    def close(self):
        """Close file handle."""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None

    def write(self, data: Dict[str, Any]):
        """
        Write a single item to file.

        Args:
            data: Data dictionary to write
        """
        if self.file_handle is None:
            self.open()

        # Write as JSONL format
        self.file_handle.write(json.dumps(data) + "\n")

        if self.auto_flush:
            self.file_handle.flush()

    def write_batch(self, data_list: List[Dict[str, Any]]):
        """
        Write multiple items to file.

        Args:
            data_list: List of data dictionaries to write
        """
        for data in data_list:
            self.write(data)

    def load_existing(self) -> List[Dict[str, Any]]:
        """
        Load all existing items from the file.

        Returns:
            List of dictionaries from the file
        """
        if not self.output_path.exists():
            return []

        items = []
        with open(self.output_path, "r") as f:
            for line in f:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return items

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_openai())