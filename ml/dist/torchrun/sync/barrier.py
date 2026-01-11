import os
import time
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

    # 3. Initialize process group
    dist.init_process_group(backend=backend)

    print(f"Process {global_rank} (local rank {local_rank}) running on {device}")

    # 4. Barrier example: synchronize all processes at a specific point

    # Different processes will take different amounts of time to reach the barrier
    # Simulate different workloads
    sleep_time = global_rank * 0.5
    print(f"Process {global_rank} will sleep for {sleep_time:.1f} seconds before barrier")
    time.sleep(sleep_time)

    print(f"Process {global_rank} reached barrier, waiting for others...")

    # Barrier: all processes wait here until everyone arrives
    dist.barrier()
    print(f"Process {global_rank} passed barrier - all processes are synchronized!")

    # 5. Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
