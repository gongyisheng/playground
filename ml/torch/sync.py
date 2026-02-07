import time

import torch

assert torch.cuda.is_available(), "CUDA required"

DEVICE = torch.device("cuda")
N = 50_000_000  # 50M floats ≈ 200 MB


def bench(label, fn, repeats=5):
    """Warmup once, then time `repeats` runs."""
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    print(f"  {label}: {elapsed*1000:.1f} ms")


# --- pin_memory ---
# Pinned (page-locked) CPU memory can't be swapped to disk,
# so the GPU DMA engine can transfer directly without a staging copy.

def test_pinned_vs_unpinned():
    gpu_tensor = torch.randn(N, device=DEVICE)

    cpu_unpinned = torch.empty(N, pin_memory=False)
    cpu_pinned = torch.empty(N, pin_memory=True)

    print("GPU -> CPU copy:")
    bench("unpinned", lambda: cpu_unpinned.copy_(gpu_tensor))
    bench("pinned  ", lambda: cpu_pinned.copy_(gpu_tensor))

    print("CPU -> GPU copy:")
    gpu_dst = torch.empty(N, device=DEVICE)
    bench("unpinned", lambda: gpu_dst.copy_(cpu_unpinned))
    bench("pinned  ", lambda: gpu_dst.copy_(cpu_pinned))


# --- non_blocking ---
# non_blocking=True queues the transfer and returns immediately,
# letting the CPU do useful work while the DMA engine runs.
# Requires pinned memory to be safe (unpinned may silently sync).

def test_copy_non_blocking():
    gpu_tensor = torch.randn(N, device=DEVICE)
    cpu_pinned = torch.empty(N, pin_memory=True)

    # CPU work that takes roughly as long as the DMA transfer (~20-40 ms)
    cpu_work_iters = 2_000_000

    def blocking_copy():
        cpu_pinned.copy_(gpu_tensor)        # blocks until copy finishes
        sum(range(cpu_work_iters))           # CPU work starts AFTER copy

    def non_blocking_copy():
        cpu_pinned.copy_(gpu_tensor, non_blocking=True)  # returns immediately
        sum(range(cpu_work_iters))           # CPU work overlaps with copy
        torch.cuda.synchronize()             # wait for transfer to finish

    print("10 transfers (GPU -> pinned CPU):")
    bench("blocking    ", blocking_copy)
    bench("non_blocking", non_blocking_copy)


# --- typical backup pattern ---
# Allocate pinned buffers once, reuse them for repeated snapshots.

def test_backup_pattern():
    params = {f"layer_{i}": torch.randn(1000, 1000, device=DEVICE) for i in range(10)}
    backup = {}

    def snapshot():
        for name, param in params.items():
            if name not in backup:
                backup[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            backup[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    def restore():
        for name, param in params.items():
            param.copy_(backup[name], non_blocking=True)
        torch.cuda.synchronize()

    snapshot()
    print("Backup/restore 10 x [1000,1000] params:")
    bench("snapshot", snapshot)
    bench("restore ", restore)


if __name__ == "__main__":
    print("=== Pinned vs Unpinned ===")
    test_pinned_vs_unpinned()
    print()
    print("=== Copy non_blocking ===")
    test_copy_non_blocking()
    print()
    print("=== Backup ===")
    test_backup_pattern()
