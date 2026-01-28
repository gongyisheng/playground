import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from prompt import SFT_PROMPT
from utils import DataWriter


@dataclass
class EvalConfig:
    model_path: str = "./outputs/qwen3-0.6b-sft/checkpoint-1300"
    input_path: str = "outputs/random_tasks.jsonl"
    output_path: str = "outputs/eval_result.jsonl"
    max_samples: Optional[int] = None  # Limit number of tasks for debugging


def compute_kl_divergence(probs: np.ndarray) -> float:
    """Compute KL divergence from uniform distribution. Lower is better."""
    probs = np.array(probs)
    probs = probs / probs.sum()  # Normalize
    n = len(probs)
    uniform = np.ones(n) / n
    # KL(uniform || model) - how much info lost when using model instead of uniform
    # Use scipy's entropy which handles edge cases
    kl = np.sum(uniform * np.log(uniform / (probs + 1e-10)))
    return float(kl)


def compute_normalized_entropy(probs: np.ndarray) -> float:
    """Compute normalized entropy. 1.0 = perfectly uniform."""
    probs = np.array(probs)
    probs = probs / probs.sum()  # Normalize
    n = len(probs)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(n)
    return float(entropy / max_entropy)


def compute_total_variation(probs: np.ndarray) -> float:
    """Compute total variation distance from uniform. Lower is better."""
    probs = np.array(probs)
    probs = probs / probs.sum()  # Normalize
    n = len(probs)
    uniform = np.ones(n) / n
    return float(0.5 * np.abs(probs - uniform).sum())


def compute_prob_ratio(probs: np.ndarray) -> float:
    """Compute ratio of max to min probability. 1.0 = perfectly uniform."""
    probs = np.array(probs)
    return float(probs.max() / (probs.min() + 1e-10))


def compute_metrics_for_task(probs: list) -> dict:
    """Compute all randomness metrics for a single task."""
    probs = np.array(probs)
    return {
        "kl_divergence": compute_kl_divergence(probs),
        "normalized_entropy": compute_normalized_entropy(probs),
        "total_variation": compute_total_variation(probs),
        "prob_ratio": compute_prob_ratio(probs),
        "max_prob": float(probs.max()),
        "min_prob": float(probs.min()),
        "std_dev": float(probs.std()),
        "num_options": len(probs),
    }


def aggregate_metrics(all_task_metrics: list) -> dict:
    """Aggregate metrics across all tasks."""
    if not all_task_metrics:
        return {}

    metric_keys = ["kl_divergence", "normalized_entropy", "total_variation",
                   "prob_ratio", "std_dev"]

    aggregated = {}
    for key in metric_keys:
        values = [m[key] for m in all_task_metrics]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return aggregated


def print_metrics_summary(aggregated: dict, all_task_metrics: list):
    """Print a formatted summary of metrics."""
    print("\n" + "=" * 60)
    print("RANDOMNESS QUALITY METRICS")
    print("=" * 60)

    for key in ["kl_divergence", "normalized_entropy", "total_variation", "prob_ratio", "std_dev"]:
        if key in aggregated:
            m = aggregated[key]
            print(f"\n{key}:")
            print(f"  Mean: {m['mean']:.4f}")
            print(f"  Std:  {m['std']:.4f}")
            print(f"  Min:  {m['min']:.4f}")
            print(f"  Max:  {m['max']:.4f}")

    # Show worst tasks
    print("\n" + "=" * 60)
    print("WORST TASKS (highest KL divergence):")
    print("=" * 60)
    sorted_metrics = sorted(all_task_metrics, key=lambda x: x["kl_divergence"], reverse=True)
    for m in sorted_metrics[:5]:
        print(f"  KL={m['kl_divergence']:.4f} | {m['task'][:50]}")

    print("\n" + "=" * 60)
    print("BEST TASKS (lowest KL divergence):")
    print("=" * 60)
    for m in sorted_metrics[-5:]:
        print(f"  KL={m['kl_divergence']:.4f} | {m['task'][:50]}")


def run_evaluation(config: EvalConfig):
    """Run full evaluation pipeline."""
    # Load model and tokenizer
    print(f"Loading model from {config.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model = model.to("cuda", dtype=torch.bfloat16)
    model.eval()

    # Load input data
    print(f"Loading data from {config.input_path}...")
    data = []
    with open(config.input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # Limit samples for debugging
    if config.max_samples is not None:
        data = data[:config.max_samples]
        print(f"Limited to {config.max_samples} samples for debugging")

    print(f"Evaluating {len(data)} examples...")

    # Store results per task for metric computation
    task_results = defaultdict(lambda: {"options": [], "probs": [], "task": ""})

    # Evaluate probabilities for each option
    with DataWriter(config.output_path, mode='w') as writer:
        for record in tqdm(data):
            task = record['task']
            options = record['options']
            uuid = record.get("uuid", task)  # Fallback to task if no uuid

            # Format prompt
            options_str = ", ".join(options)
            prompt_text = SFT_PROMPT.format(task=task, options=options_str)

            # Loop through each option and calculate its probability
            for option in options:
                messages = [
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": option},
                ]

                # Apply chat template for full text
                full_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                # Tokenize full text
                full_encoding = tokenizer(full_text, return_tensors="pt").to("cuda")
                input_ids = full_encoding['input_ids']

                # Tokenize user text to find where assistant starts
                user_text = tokenizer.apply_chat_template(
                    messages[:1],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                user_encoding = tokenizer(user_text, return_tensors="pt")
                user_length = user_encoding['input_ids'].shape[1]

                # Tokenize assistant tokens
                assistant_tokens = tokenizer.encode(option + tokenizer.eos_token, add_special_tokens=False)
                assistant_length = len(assistant_tokens)

                # Forward pass
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits

                # Calculate probability for assistant tokens
                log_probs = []
                for i in range(assistant_length):
                    token_idx = user_length + i - 1  # -1 because logits are shifted
                    target_token_id = assistant_tokens[i]

                    token_logits = logits[0, token_idx, :]
                    token_log_probs = F.log_softmax(token_logits, dim=-1)
                    log_prob = token_log_probs[target_token_id].item()
                    log_probs.append(log_prob)

                # Calculate total probability (sum of log probs = log of product of probs)
                total_log_prob = sum(log_probs)
                total_proba = torch.exp(torch.tensor(total_log_prob)).item()

                # Store for metrics
                task_results[uuid]["options"].append(option)
                task_results[uuid]["probs"].append(total_proba)
                task_results[uuid]["task"] = task

                # Save one line per option
                result = {
                    "uuid": uuid,
                    "task": task,
                    "options": options,
                    "chosen_option": option,
                    "proba": total_proba
                }
                writer.write(result)

    print(f"Results saved to {config.output_path}")

    # Compute metrics for each task
    all_task_metrics = []
    for uuid, data in task_results.items():
        metrics = compute_metrics_for_task(data["probs"])
        metrics["task"] = data["task"]
        metrics["uuid"] = uuid
        all_task_metrics.append(metrics)

    # Aggregate and print
    aggregated = aggregate_metrics(all_task_metrics)
    print_metrics_summary(aggregated, all_task_metrics)

    return aggregated, all_task_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./outputs/qwen3-0.6b-sft/checkpoint-1300")
    parser.add_argument("--input_path", type=str, default="outputs/random_tasks.jsonl")
    parser.add_argument("--output_path", type=str, default="outputs/eval_result.jsonl")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for debugging")
    args = parser.parse_args()

    config = EvalConfig(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
    )

    run_evaluation(config)
