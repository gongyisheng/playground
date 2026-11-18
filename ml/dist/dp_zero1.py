"""Minimal ZeRO Stage 1 data parallel training, using only PyTorch.

Run on two CPU workers:
    torchrun --standalone --nproc-per-node=2 zero1_demo.py

Run on two GPUs (one GPU per worker):
    torchrun --standalone --nproc-per-node=2 zero1_demo.py --device cuda

Each worker keeps the full model and averaged gradients, but AdamW only owns
optimizer state for its assigned parameters. This demo assigns whole tensors
and uses synchronous collectives; production implementations pack/bucket data.
The model has four equally sized parameter tensors, supporting 1-4 workers.
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def make_batch(rank, batch_size, device):
    generator = torch.Generator(device=device).manual_seed(1000 + rank)
    x = torch.randn(batch_size, 32, generator=generator, device=device)
    return x, x.roll(shifts=1, dims=1)


def train(args, device):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 32, bias=False),
        nn.GELU(),
        nn.Linear(32, 32, bias=False),
        nn.GELU(),
        nn.Linear(32, 32, bias=False),
        nn.GELU(),
        nn.Linear(32, 32, bias=False),
    ).to(device)
    params = list(model.parameters())
    if world_size > len(params):
        raise ValueError("Use 1-4 workers, or add more parameter tensors.")

    # Begin with identical replicas, even if initialization seeds differ.
    with torch.no_grad():
        for param in params:
            dist.broadcast(param, src=0)

    # Only the owner creates Adam's first/second moment buffers for a tensor.
    owners = [i % world_size for i in range(len(params))]
    owned = [p for p, owner in zip(params, owners) if owner == rank]
    optimizer = torch.optim.AdamW(owned, lr=0.01, weight_decay=0.01, foreach=False)
    x, y = make_batch(rank, args.batch_size, device)

    if rank == 0:
        print(f"ZeRO-1 demo: {world_size} workers, device={device.type}", flush=True)

    for step in range(args.steps):
        # Clear ALL gradients: optimizer.zero_grad() would clear only our shard.
        model.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()

        # 1. Replicate the averaged gradients, as in ordinary data parallelism.
        # Equal local batch sizes make this the global mean-loss gradient.
        for param in params:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

        # 2. Update only this worker's parameters using its local AdamW state.
        optimizer.step()

        # 3. Replicate the updated parameters from their owners to every worker.
        with torch.no_grad():
            for param, owner in zip(params, owners):
                dist.broadcast(param, src=owner)

        mean_loss = loss.detach().clone()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(
                f"step {step + 1:02d}: loss={mean_loss.item() / world_size:.6f}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    if "RANK" not in os.environ:
        parser.error("Launch with torchrun; see the examples in --help")

    device = torch.device(args.device)
    if args.device == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    try:
        train(args, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
