from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import json


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from file (supports parquet, json, jsonl, csv)."""
    input_path = Path(file_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    suffix = input_path.suffix.lower()

    if suffix == '.parquet':
        df = pd.read_parquet(input_path)
    elif suffix == '.csv':
        df = pd.read_csv(input_path)
    elif suffix == '.json':
        df = pd.read_json(input_path)
    elif suffix == '.jsonl':
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: .parquet, .json, .jsonl, .csv")

    return df


def validate_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """Validate that all required columns exist in dataset and convert to strings."""
    required_columns = set(column_mapping.values())
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    # Validate that all required columns are strings or can be converted to strings
    df = df.copy()
    for col in required_columns:
        if not pd.api.types.is_string_dtype(df[col]):
            # Try to convert to string
            try:
                df[col] = df[col].astype(str)
            except Exception as e:
                raise ValueError(f"Column '{col}' cannot be converted to string: {e}")

    return df


def prepare_template_kwargs(row: pd.Series, column_mapping: Dict[str, str]) -> Dict[str, str]:
    """Prepare kwargs for template formatting from a dataset row."""
    kwargs = {}
    for template_var, column_name in column_mapping.items():
        kwargs[template_var] = row[column_name]
    return kwargs


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to file (supports parquet, json, jsonl, csv)."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Determine format from file extension
    suffix = output_path.suffix.lower()

    if suffix == '.parquet':
        df.to_parquet(output_path, index=False)
    elif suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif suffix == '.json':
        df.to_json(output_path, orient='records', indent=2)
    elif suffix == '.jsonl':
        df.to_json(output_path, orient='records', lines=True)
    else:
        raise ValueError(f"Unsupported output format: {suffix}. Supported: .parquet, .json, .jsonl, .csv")
