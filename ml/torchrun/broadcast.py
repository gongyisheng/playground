import os
import torch
import torch.distributed as dist

# BROADCAST (src=0):
#     Rank0: [42, 43] ──> Rank0: [42, 43]
#                     ──> Rank1: [42, 43]
#                     ──> Rank2: [42, 43]

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

    # 3. Initialize process group
    dist.init_process_group(backend=backend)

    print(f"Process {global_rank} (local rank {local_rank}) running on {device}")

    # 4. Broadcast example: one process (source) sends its tensor to all other processes

    # Define the source rank (rank 0 will broadcast its data)
    source_rank = 0

    if global_rank == source_rank:
        # Source process creates data to broadcast
        tensor = torch.tensor([100.0, 200.0, 300.0], device=device)
        print(f"Process {global_rank} (SOURCE) broadcasting tensor: {tensor}")
    else:
        # Other processes create empty tensor to receive data
        tensor = torch.zeros(3, device=device)
        print(f"Process {global_rank} before broadcast: {tensor}")

    # Broadcast from source to all processes
    dist.broadcast(tensor, src=source_rank)

    print(f"Process {global_rank} after broadcast: {tensor}")

    # 5. Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
