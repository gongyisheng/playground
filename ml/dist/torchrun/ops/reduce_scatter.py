import os
import torch
import torch.distributed as dist

# REDUCE_SCATTER:
#     Rank0: [[1], [2], [3]]  ─┐    Rank0 gets [6]  (1+2+3)
#     Rank1: [[2], [3], [4]]  ─┼─>  Rank1 gets [9]  (2+3+4)
#     Rank2: [[3], [4], [5]]  ─┘    Rank2 gets [12] (3+4+5)

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

    # 4. Reduce-scatter example: reduce tensors and scatter result
    # Each process starts with a tensor of size world_size
    # The value is (global_rank + 1) for easy verification
    input_tensor = torch.full((world_size,), global_rank + 1, dtype=torch.float32).to(device)
    print(f"Process {global_rank} input: {input_tensor.tolist()}")

    # Output tensor will hold this process's portion of the reduced result
    output_tensor = torch.zeros(1).to(device)

    # Reduce-scatter: reduces across all processes, then each process gets a slice
    dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)

    # Expected result: sum of (rank + 1) for all ranks = 1 + 2 + ... + world_size
    expected = world_size * (world_size + 1) // 2
    print(f"Process {global_rank} after reduce_scatter: {output_tensor.item()} (expected: {expected})")

    # 5. Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
