import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Init NCCL process group
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    # Set GPU for this rank
    torch.cuda.set_device(rank)

    # Each GPU starts with a different value (rank + 1), helps see if sum is correct
    x = torch.tensor([rank + 1.0], device=rank)

    # Perform an all-reduce (sum)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # Print result on each rank
    print(f"[Rank {rank}] After all_reduce: {x.item()}")

    # Sync and close
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
