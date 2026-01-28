import os
import torch
import torch.distributed as dist

# similar to ALL_GATHER:
#     Rank0: [0]  ─┐
#     Rank1: [10] ─┼─> All ranks get [[0], [10], [20]]
#     Rank2: [20] ─┘
# use dist.gather_object() to just store one replica for specific process

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

    # On each rank
    rank = dist.get_rank()
    my_data = {"rank": rank, "values": [rank * 10, rank * 20]}
    gathered_list = [None] * dist.get_world_size()

    dist.all_gather_object(gathered_list, my_data)
    print(f"Rank {rank}: {gathered_list}")
    
    # 5. Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()