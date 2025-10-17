"""
Benchmark utilities and statistics
"""

import time
import numpy as np
from typing import List
from dataclasses import dataclass
import torch


@dataclass
class BenchmarkResult:
    method_name: str
    avg_latency_ms: float
    throughput_samples_per_sec: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float


def benchmark_service(service):
    print("BERT Embedding Inference Speed Benchmark")
    print("Model: bert-base-uncased")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*100)

    # Test data
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
    ]

    num_iterations = 100
    results = []

    # 1. Base Transformers
    print("\n1. Testing Base Transformers Method...")
    try:
        # Warmup
        print(f"  Warming up...")
        service.warmup(num_runs=10)

        # Benchmark
        latencies = []
        print(f"  Running {num_iterations} iterations...")

        for i in range(num_iterations):
            start_time = time.perf_counter()
            _ = service.encode(test_texts)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{num_iterations}")

        # Calculate statistics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        throughput = 1000 / avg_latency * len(test_texts)

        result = BenchmarkResult(
            method_name=service.__class__.__name__,
            avg_latency_ms=avg_latency,
            throughput_samples_per_sec=throughput,
            std_latency_ms=std_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency
        )
        results.append(result)
        print(f"   ✓ Completed: {result.avg_latency_ms:.2f} ms")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Print results
    if results:
        print_results(results)
    else:
        print("\nNo successful benchmarks to report.")


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table"""

    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    print(f"{'Method':<30} {'Avg Latency (ms)':<20} {'Throughput (samples/s)':<25} {'Std (ms)':<15}")
    print("-"*100)

    # Sort by average latency
    results_sorted = sorted(results, key=lambda x: x.avg_latency_ms)
    baseline = results_sorted[-1].avg_latency_ms  # Slowest method

    for result in results_sorted:
        speedup = baseline / result.avg_latency_ms
        print(f"{result.method_name:<30} {result.avg_latency_ms:>10.2f} ({speedup:.2f}x) {result.throughput_samples_per_sec:>15.2f} {result.std_latency_ms:>15.2f}")

    print("-"*100)
    print(f"\nDetailed Statistics:")
    for result in results_sorted:
        print(f"\n{result.method_name}:")
        print(f"  Average: {result.avg_latency_ms:.2f} ms")
        print(f"  Std Dev: {result.std_latency_ms:.2f} ms")
        print(f"  Min: {result.min_latency_ms:.2f} ms")
        print(f"  Max: {result.max_latency_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")
    print("="*100)
