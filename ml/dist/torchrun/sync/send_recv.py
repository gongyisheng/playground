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

    # 3. Initialize process group
    dist.init_process_group(backend=backend)

    print(f"Process {global_rank} (local rank {local_rank}) running on {device}")

    # 4. Send/Recv example: point-to-point communication
    # This is commonly used in pipeline parallelism where each stage
    # sends activations to the next stage and receives gradients back

    if world_size < 2:
        print("send_recv requires at least 2 processes. Run with: torchrun --nproc_per_node=2 send_recv.py")
        dist.destroy_process_group()
        return

    # Example 1: Simple pipeline-style communication (rank 0 -> rank 1 -> rank 2 -> ...)
    if global_rank == 0:
        # First process: send data to next process
        tensor = torch.tensor([100.0, 200.0, 300.0], device=device)
        print(f"Rank {global_rank} sending tensor to rank 1: {tensor}")
        dist.send(tensor, dst=1)

    elif global_rank == world_size - 1:
        # Last process: only receive from previous process
        tensor = torch.zeros(3, device=device)
        dist.recv(tensor, src=global_rank - 1)
        print(f"Rank {global_rank} received tensor from rank {global_rank - 1}: {tensor}")

    else:
        # Middle processes: receive from previous, process, send to next
        tensor = torch.zeros(3, device=device)
        dist.recv(tensor, src=global_rank - 1)
        print(f"Rank {global_rank} received tensor from rank {global_rank - 1}: {tensor}")

        # Process the data (e.g., add rank to simulate computation)
        tensor += global_rank
        print(f"Rank {global_rank} processed tensor: {tensor}")
        print(f"Rank {global_rank} sending tensor to rank {global_rank + 1}")
        dist.send(tensor, dst=global_rank + 1)

    # Synchronize before next example
    dist.barrier()

    # Example 2: Bidirectional communication (useful for pipeline parallelism with gradients)
    if global_rank == 0:
        # Send forward pass data
        forward_tensor = torch.tensor([10.0, 20.0], device=device)
        print(f"\nRank {global_rank} sending forward pass to rank 1: {forward_tensor}")
        dist.send(forward_tensor, dst=1)

        # Receive backward pass gradient
        backward_tensor = torch.zeros(2, device=device)
        dist.recv(backward_tensor, src=1)
        print(f"Rank {global_rank} received backward pass from rank 1: {backward_tensor}")

    elif global_rank == 1:
        # Receive forward pass data
        forward_tensor = torch.zeros(2, device=device)
        dist.recv(forward_tensor, src=0)
        print(f"Rank {global_rank} received forward pass from rank 0: {forward_tensor}")

        # Send backward pass gradient
        backward_tensor = forward_tensor * 2  # Simulate gradient
        print(f"Rank {global_rank} sending backward pass to rank 0: {backward_tensor}")
        dist.send(backward_tensor, dst=0)

    # 5. Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
