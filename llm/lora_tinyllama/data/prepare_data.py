#!/usr/bin/env python3
"""
Data preparation script for TinyLlama fine-tuning
This script can be used to prepare custom datasets or explore existing ones
"""

import os
import sys
import json
import argparse
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_custom_dataset(file_path: str, format_type: str = "auto") -> Dataset:
    """
    Load a custom dataset from file
    
    Args:
        file_path: Path to the dataset file
        format_type: Format of the file (auto, json, jsonl, csv)
        
    Returns:
        HuggingFace Dataset object
    """
    if format_type == "auto":
        ext = os.path.splitext(file_path)[1].lower()
        format_type = ext[1:] if ext else "json"
    
    logger.info(f"Loading dataset from {file_path} as {format_type}")
    
    if format_type == "json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
        
    elif format_type == "jsonl":
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        dataset = Dataset.from_list(data)
        
    elif format_type == "csv":
        df = pd.read_csv(file_path)
        dataset = Dataset.from_pandas(df)
        
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    return dataset


def convert_to_chat_format(
    dataset: Dataset,
    instruction_col: str = "instruction",
    input_col: str = "input",
    output_col: str = "output",
    system_message: str = "You are a helpful Python coding assistant."
) -> Dataset:
    """
    Convert dataset to chat format
    
    Args:
        dataset: Input dataset
        instruction_col: Column name for instruction
        input_col: Column name for input (optional)
        output_col: Column name for output
        system_message: System message to add
        
    Returns:
        Dataset with chat format
    """
    def format_example(example):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": example[instruction_col]}
        ]
        
        if input_col in example and example[input_col]:
            messages[1]["content"] += f"\n\nInput: {example[input_col]}"
        
        messages.append({"role": "assistant", "content": example[output_col]})
        
        return {"messages": messages}
    
    logger.info("Converting dataset to chat format...")
    return dataset.map(format_example)


def analyze_dataset(dataset: Dataset, tokenizer_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> Dict[str, Any]:
    """
    Analyze dataset statistics
    
    Args:
        dataset: Dataset to analyze
        tokenizer_name: Tokenizer to use for token counting
        
    Returns:
        Dictionary with statistics
    """
    logger.info("Analyzing dataset...")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    stats = {
        "num_examples": len(dataset),
        "columns": dataset.column_names,
        "sample_examples": []
    }
    
    # Analyze text lengths
    if "text" in dataset.column_names:
        texts = dataset["text"][:100]  # Sample first 100
        token_lengths = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            token_lengths.append(len(tokens))
        
        stats["avg_tokens"] = sum(token_lengths) / len(token_lengths)
        stats["max_tokens"] = max(token_lengths)
        stats["min_tokens"] = min(token_lengths)
        
    # Get sample examples
    for i in range(min(3, len(dataset))):
        stats["sample_examples"].append(dataset[i])
    
    return stats


def create_sample_dataset(output_path: str, num_examples: int = 100):
    """
    Create a sample Python instruction dataset
    
    Args:
        output_path: Where to save the dataset
        num_examples: Number of examples to create
    """
    logger.info(f"Creating sample dataset with {num_examples} examples...")
    
    examples = []
    
    # Python coding examples
    templates = [
        {
            "instruction": "Write a Python function to calculate the factorial of a number",
            "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)"
        },
        {
            "instruction": "How do I read a CSV file in Python?",
            "output": "You can read a CSV file using the pandas library:\n\nimport pandas as pd\ndf = pd.read_csv('file.csv')\n\nOr using the csv module:\n\nimport csv\nwith open('file.csv', 'r') as file:\n    reader = csv.reader(file)\n    for row in reader:\n        print(row)"
        },
        {
            "instruction": "Explain list comprehension in Python",
            "output": "List comprehension is a concise way to create lists in Python. It consists of brackets containing an expression followed by a for clause.\n\nSyntax: [expression for item in iterable if condition]\n\nExample:\n# Create a list of squares\nsquares = [x**2 for x in range(10)]\n\n# Filter even numbers\nevens = [x for x in range(20) if x % 2 == 0]"
        },
        {
            "instruction": "Write a function to check if a string is a palindrome",
            "output": "def is_palindrome(s):\n    # Remove spaces and convert to lowercase\n    s = s.replace(' ', '').lower()\n    # Check if string equals its reverse\n    return s == s[::-1]\n\n# Example usage:\nprint(is_palindrome('racecar'))  # True\nprint(is_palindrome('hello'))    # False"
        },
        {
            "instruction": "How do I handle exceptions in Python?",
            "output": "Use try-except blocks to handle exceptions:\n\ntry:\n    # Code that might raise an exception\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Cannot divide by zero')\nexcept Exception as e:\n    print(f'An error occurred: {e}')\nfinally:\n    print('This always executes')"
        }
    ]
    
    # Duplicate and vary examples to reach num_examples
    while len(examples) < num_examples:
        for template in templates:
            if len(examples) >= num_examples:
                break
            examples.append(template.copy())
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    
    logger.info(f"Sample dataset saved to {output_path}")


def explore_dataset(dataset_name: str):
    """
    Explore a HuggingFace dataset
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
    """
    logger.info(f"Exploring dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    print("\n" + "="*50)
    print(f"Dataset: {dataset_name}")
    print("="*50)
    
    # Print dataset info
    print(f"\nDataset structure:")
    print(dataset)
    
    # Print splits
    print(f"\nAvailable splits: {list(dataset.keys())}")
    
    # Analyze train split if available
    if "train" in dataset:
        train_data = dataset["train"]
        print(f"\nTrain split size: {len(train_data)}")
        print(f"Columns: {train_data.column_names}")
        
        # Show sample examples
        print("\nSample examples:")
        for i in range(min(3, len(train_data))):
            print(f"\n--- Example {i+1} ---")
            example = train_data[i]
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                print(f"{key}: {value}")
    
    print("\n" + "="*50)


def main():
    parser = argparse.ArgumentParser(description="Data preparation utilities")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Explore dataset command
    explore_parser = subparsers.add_parser("explore", help="Explore a dataset")
    explore_parser.add_argument("dataset", help="Dataset name on HuggingFace")
    
    # Create sample dataset command
    sample_parser = subparsers.add_parser("sample", help="Create a sample dataset")
    sample_parser.add_argument(
        "--output",
        default="sample_python_dataset.json",
        help="Output file path"
    )
    sample_parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to create"
    )
    
    # Convert dataset command
    convert_parser = subparsers.add_parser("convert", help="Convert dataset to chat format")
    convert_parser.add_argument("input", help="Input dataset file")
    convert_parser.add_argument("output", help="Output dataset file")
    convert_parser.add_argument(
        "--instruction-col",
        default="instruction",
        help="Column name for instruction"
    )
    convert_parser.add_argument(
        "--output-col",
        default="output",
        help="Column name for output"
    )
    
    # Analyze dataset command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze dataset statistics")
    analyze_parser.add_argument("dataset", help="Dataset file or HuggingFace name")
    
    args = parser.parse_args()
    
    if args.command == "explore":
        explore_dataset(args.dataset)
        
    elif args.command == "sample":
        create_sample_dataset(args.output, args.num_examples)
        
    elif args.command == "convert":
        dataset = load_custom_dataset(args.input)
        converted = convert_to_chat_format(
            dataset,
            instruction_col=args.instruction_col,
            output_col=args.output_col
        )
        converted.save_to_disk(args.output)
        logger.info(f"Converted dataset saved to {args.output}")
        
    elif args.command == "analyze":
        if os.path.exists(args.dataset):
            dataset = load_custom_dataset(args.dataset)
        else:
            dataset = load_dataset(args.dataset)["train"]
        
        stats = analyze_dataset(dataset)
        print("\nDataset Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()