import os
import torch
import torch.distributed as dist

def main():
    # 1. Get rank information (set by torchrun)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 2. Check CUDA availability and set device
    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    # 3. Initialize process group (allows processes to communicate)
    # Use 'nccl' backend for CUDA (fastest for GPU communication)
    # Use 'gloo' backend for CPU
    dist.init_process_group(backend=backend)

    print(f"Process {global_rank} (local rank {local_rank}) running on {device}")

    # 4. All-reduce example: sum tensors from all processes
    # Each process starts with a tensor of ones
    tensor = torch.ones(1).to(device)
    print(f"Process {global_rank} before all_reduce: {tensor.item()}")

    # All-reduce with SUM operation: each process will end up with the sum
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Process {global_rank} after all_reduce: {tensor.item()} (expected: {world_size})")

    # 5. Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
