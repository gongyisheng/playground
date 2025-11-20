import json
from datasets import load_dataset


def map_fn(example):
    """
    Process each row of the dataset.
    Parses the JSON string containing conversation messages.

    Args:
        example: A dataset row with conversation data

    Returns:
        Dictionary with parsed messages
    """
    return {"messages": json.loads(example['messages'])}


def load_train_dataset(dataset_path: str):
    """Load the training dataset from parquet file"""
    dataset = load_dataset("parquet", data_files=dataset_path, split="train")
    dataset = dataset.map(map_fn)
    return dataset