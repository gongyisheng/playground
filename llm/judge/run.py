"""Run batch judging from a YAML configuration file."""
import asyncio
from typing import Dict, Any

from tqdm import tqdm

from judge import LLMJudge
from config import Config
from dataset import load_dataset, validate_columns, save_results, prepare_template_kwargs


async def judge_dataset(judge: LLMJudge, input_file: str, output_file: str,
                       column_mapping: Dict[str, str],
                       judge_kwargs: Dict[str, Any],
                       batch_size: int = 32, retry: int = 3):
    """
    Judge an entire dataset from file.

    Args:
        judge: LLMJudge instance
        input_file: Path to input file (.csv, .json, .jsonl, .parquet)
        output_file: Path to output file (.csv, .json, .jsonl, .parquet)
        column_mapping: Map template variables to dataset columns
        judge_kwargs: Additional kwargs to pass to judge method (method, min_score, max_score, etc.)
        batch_size: Number of concurrent requests
        retry: Number of retry attempts
    """
    print(f"Loading dataset from {input_file}...")
    df = load_dataset(input_file)
    print(f"Loaded {len(df)} rows")

    df = validate_columns(df, column_mapping)
    print(f"Validated columns: {list(column_mapping.values())}")

    # Prepare items from DataFrame
    print(f"\nPreparing items for judging...")
    items = []
    for _, row in df.iterrows():
        # Get template kwargs for this row
        template_kwargs = prepare_template_kwargs(row, column_mapping)
        # Include original row data + template_kwargs (but NOT judge_kwargs)
        item = {**row.to_dict(), **template_kwargs}
        items.append(item)

    # Process dataset in batches
    print(f"Processing {len(items)} items in batches (batch_size={batch_size})...")
    all_results = []

    with tqdm(total=len(items), desc="Judging") as pbar:
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            batch_results = await judge.judge_batch(batch_items, retry, judge_kwargs)
            all_results.extend(batch_results)
            pbar.update(len(batch_items))

    results = all_results

    # Save results and print statistics
    print(f"\nSaving results to {output_file}...")
    save_results(results, output_file)


async def main(config_path):
    # Load configuration
    config = Config.from_yaml(config_path)

    # Create judge
    judge = LLMJudge(
        prompt_template=config.judge.prompt,
        provider=config.judge.provider,
        model=config.judge.model
    )

    # Prepare judge kwargs based on method
    judge_kwargs = {
        'method': config.judge.method,
        'min_score': config.judge.min_score,
        'max_score': config.judge.max_score,
    }

    # Add method-specific parameters
    if config.judge.method == 'monte_carlo' and config.judge.monte_carlo:
        judge_kwargs['temperature'] = config.judge.monte_carlo.temperature
        judge_kwargs['max_tokens'] = config.judge.monte_carlo.max_tokens
        judge_kwargs['monte_carlo_num_rounds'] = config.judge.monte_carlo.num_rounds
        judge_kwargs['monte_carlo_score_pattern'] = config.judge.monte_carlo.score_pattern
    elif config.judge.method == 'logprob_weighted' and config.judge.logprob_weighted:
        judge_kwargs['temperature'] = config.judge.logprob_weighted.temperature
        judge_kwargs['max_tokens'] = config.judge.logprob_weighted.max_tokens

    # Run batch judging
    await judge_dataset(
        judge=judge,
        input_file=config.dataset.input_file,
        output_file=config.dataset.output_file,
        column_mapping=config.dataset.column_mapping,
        judge_kwargs=judge_kwargs,
        batch_size=config.judge.batch_size,
        retry=config.judge.retry
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python run.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    asyncio.run(main(config_path))
