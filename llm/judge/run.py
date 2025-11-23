"""Run batch judging from a YAML configuration file."""
import asyncio
from typing import Dict

from tqdm import tqdm

from judge import LLMJudge
from config import JudgeConfig
from dataset import load_dataset, validate_columns, save_results, prepare_template_kwargs


async def judge_dataset(judge: LLMJudge, input_file: str, output_file: str,
                       column_mapping: Dict[str, str],
                       batch_size: int = 32, retry: int = 3):
    """
    Judge an entire dataset from file.

    Args:
        judge: LLMJudge instance
        input_file: Path to input file (.csv, .json, .jsonl, .parquet)
        output_file: Path to output file (.csv, .json, .jsonl, .parquet)
        column_mapping: Map template variables to dataset columns
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
        # Include original row data
        item = row.to_dict()
        item.update(template_kwargs)
        items.append(item)

    # Process dataset in batches
    print(f"Processing {len(items)} items in batches (batch_size={batch_size})...")
    all_results = []

    with tqdm(total=len(items), desc="Judging") as pbar:
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            batch_results = await judge.judge_batch(batch_items, retry)
            all_results.extend(batch_results)
            pbar.update(len(batch_items))

    results = all_results

    # Save results and print statistics
    print(f"\nSaving results to {output_file}...")
    save_results(results, output_file)


async def main(config_path):
    # Load configuration
    config = JudgeConfig.from_yaml(config_path)

    # Create judge
    judge = LLMJudge(
        prompt_template=config.template.prompt,
        provider=config.llm.provider,
        model=config.llm.model
    )

    # Run batch judging
    await judge_dataset(
        judge=judge,
        input_file=config.dataset.input_file,
        output_file=config.dataset.output_file,
        column_mapping=config.template.column_mapping,
        batch_size=config.processing.batch_size,
        retry=config.processing.retry
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python run.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]

    asyncio.run(main(config_path))
