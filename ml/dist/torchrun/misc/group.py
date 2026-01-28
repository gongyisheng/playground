"""
PyTorch Distributed Process Group Examples

Run with: torchrun --nproc_per_node=4 group.py
"""

import os
import torch
import torch.distributed as dist

# PROCESS GROUPS:
#     WORLD group: all ranks participate
#     Subgroup: only specified ranks participate
#     Backend: nccl (GPU) or gloo (CPU)


def main():
    # Get rank information (set by torchrun)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Check CUDA availability and set device
    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    # Initialize process group
    dist.init_process_group(backend=backend)

    print(f"Process {global_rank} (local rank {local_rank}) running on {device}")

    # Default World Group
    world_group = dist.group.WORLD
    print(f"[Rank {global_rank}] World group size: {dist.get_world_size(world_group)}")

    # Create Even/Odd Groups
    even_ranks = [r for r in range(world_size) if r % 2 == 0]
    odd_ranks = [r for r in range(world_size) if r % 2 == 1]

    even_group = dist.new_group(ranks=even_ranks)
    odd_group = dist.new_group(ranks=odd_ranks) if odd_ranks else None

    if global_rank % 2 == 0:
        print(f"[Rank {global_rank}] Member of even_group: {even_ranks}")
    else:
        print(f"[Rank {global_rank}] Member of odd_group: {odd_ranks}")

    # Using Groups with Collective Operations
    tensor = torch.tensor([global_rank + 1.0], device=device)

    # All-reduce within even group only
    if global_rank % 2 == 0:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=even_group)
        print(f"[Rank {global_rank}] Even group all_reduce result: {tensor}")

    # All-reduce within odd group only
    if global_rank % 2 == 1 and odd_group is not None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=odd_group)
        print(f"[Rank {global_rank}] Odd group all_reduce result: {tensor}")

    dist.barrier()

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
