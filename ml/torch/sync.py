import time

import torch

assert torch.cuda.is_available(), "CUDA required"


# --- .cuda()/.cpu() memory cost ---
# .cuda()/.cpu() return a NEW tensor; the original stays alive.
# So both copies coexist in memory until one is deleted/GC'd.
#
#   cpu_t = torch.randn(N)            # 200 MB CPU
#   gpu_t = cpu_t.cuda()              # +200 MB GPU  (cpu_t still alive)
#   cpu_t2 = gpu_t.cpu()              # +200 MB CPU  (gpu_t still alive)
#   # total: 400 MB CPU + 200 MB GPU
#
# .copy_() writes into an existing buffer — no extra allocation.
#   buf.copy_(gpu_t)                  # buf reused, 0 new alloc

# --- .cuda() / .cpu() vs .copy_() ---
# .cuda() and .cpu() are convenient but suboptimal for hot paths:
#   gpu_t = cpu_t.cuda()  # always allocates new tensor (no buffer reuse)
#   cpu_t = gpu_t.cpu()   # always unpinned memory (no DMA fast-path)
#                         # always blocking (no overlap with CPU work)
#
# For repeated transfers, prefer pre-allocated pinned buffers + .copy_():
#   buf = torch.empty_like(t, device="cpu", pin_memory=True)  # once
#   buf.copy_(gpu_t, non_blocking=True)                       # reuse, async
#
# .cuda()/.cpu() are fine for one-off transfers (model loading, prototyping).

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


# --- torch.cuda.empty_cache() ---
# PyTorch uses a caching allocator: del'd tensors go back to a free pool,
# NOT back to CUDA. memory_allocated() drops, but memory_reserved() stays.
# empty_cache() returns the free pool to CUDA so other processes can use it.
#
# When to use:
#   1. Before a phase that needs a big contiguous block (e.g. large batch inference after training)
#   2. When sharing GPU with other processes (notebooks, multi-model serving)
#   3. After deleting a large model before loading another
# When NOT to use:
#   Repeatedly in a training loop — the cache is there to avoid cudaMalloc overhead.

def test_empty_cache():
    # clean up 
    torch.cuda.empty_cache()

    # allocate 200 MB on GPU
    t = torch.randn(N, device=DEVICE)
    reserved_before = torch.cuda.memory_reserved()
    allocated_before = torch.cuda.memory_allocated()
    print(f"  with tensor:    allocated={allocated_before/1e6:.0f} MB  reserved={reserved_before/1e6:.0f} MB")

    # del returns memory to PyTorch's cache, NOT to CUDA
    del t
    reserved_after_del = torch.cuda.memory_reserved()
    allocated_after_del = torch.cuda.memory_allocated()
    print(f"  after del:      allocated={allocated_after_del/1e6:.0f} MB  reserved={reserved_after_del/1e6:.0f} MB  (still reserved!)")

    # empty_cache() returns cached memory to CUDA
    torch.cuda.empty_cache()
    reserved_after_cache = torch.cuda.memory_reserved()
    allocated_after_cache = torch.cuda.memory_allocated()
    print(f"  after cache():  allocated={allocated_after_cache/1e6:.0f} MB  reserved={reserved_after_cache/1e6:.0f} MB  (freed to CUDA)")


if __name__ == "__main__":
    print("=== Pinned vs Unpinned ===")
    test_pinned_vs_unpinned()
    print()
    print("=== Copy non_blocking ===")
    test_copy_non_blocking()
    print()
    print("=== Backup ===")
    test_backup_pattern()
    print()
    print("=== empty_cache() ===")
    test_empty_cache()
