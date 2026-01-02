import os
import torch
import torch.distributed as dist

# ALL_GATHER:
#     Rank0: [0]  ─┐
#     Rank1: [10] ─┼─> All ranks get [[0], [10], [20]]
#     Rank2: [20] ─┘

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

    # 4. All-gather example: each process has its own tensor,
    #    and all processes gather tensors from all other processes

    # Create a tensor unique to this process (filled with its rank)
    local_tensor = torch.full((3,), float(global_rank), device=device)
    print(f"Process {global_rank} local tensor: {local_tensor}")

    # Prepare list to receive tensors from all processes
    gathered_tensors = [torch.zeros(3, device=device) for _ in range(world_size)]

    # Gather all tensors
    dist.all_gather(gathered_tensors, local_tensor)

    print(f"Process {global_rank} gathered tensors from all processes:")
    for i, tensor in enumerate(gathered_tensors):
        print(f"  Rank {i}: {tensor}")

    # 5. Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
