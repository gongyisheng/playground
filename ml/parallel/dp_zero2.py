"""Minimal ZeRO Stage 2 data parallel training, using only PyTorch.

Run on two CPU workers:
    torchrun --standalone --nproc-per-node=2 dp_zero2.py

Run on two GPUs (one GPU per worker):
    torchrun --standalone --nproc-per-node=2 dp_zero2.py --device cuda

Each worker keeps the full model, but only a slice of each averaged gradient
and its AdamW optimizer state. Backward hooks reduce-scatter each gradient and
release the full gradient immediately. After local updates, all-gather restores
the full parameters. Tensors are padded to support uneven splits. This demo
uses synchronous collectives per tensor; production implementations use buckets
and overlap communication with computation.
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

    # Begin with identical replicas, even if initialization seeds differ.
    with torch.no_grad():
        for param in params:
            dist.broadcast(param, src=0)

    def reduce_gradient(param, shard):
        # All workers run the same graph, so hooks use the same collective order.
        flat_grad = F.pad(
            param.grad.flatten(), (0, shard.numel() * world_size - param.numel())
        )
        shard.grad = torch.empty_like(shard)
        dist.reduce_scatter_tensor(shard.grad, flat_grad, op=dist.ReduceOp.SUM)
        # Equal local batch sizes make this the global mean-loss gradient slice.
        shard.grad.div_(world_size)
        param.grad = None

    # AdamW owns only this worker's slice of each parameter and its moment buffers.
    shards = []
    for param in params:
        flat_param = F.pad(param.detach().flatten(), (0, -param.numel() % world_size))
        shard = nn.Parameter(flat_param.chunk(world_size)[rank].clone())
        shards.append(shard)
        param.register_post_accumulate_grad_hook(
            lambda param, shard=shard: reduce_gradient(param, shard)
        )
    optimizer = torch.optim.AdamW(shards, lr=0.01, weight_decay=0.01, foreach=False)
    x, y = make_batch(rank, args.batch_size, device)

    if rank == 0:
        print(f"ZeRO-2 demo: {world_size} workers, device={device.type}", flush=True)

    for step in range(args.steps):
        # Full model gradients are cleared by the hooks; clear our saved slices.
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)

        # 1. Reduce-scatter gradients during backward, keeping only local slices.
        loss.backward()

        # 2. Update local parameter slices using the sharded gradients and state.
        optimizer.step()

        # 3. Gather updated slices so every worker again has the full parameters.
        with torch.no_grad():
            for param, shard in zip(params, shards):
                flat_param = torch.empty(
                    shard.numel() * world_size, dtype=param.dtype, device=device
                )
                dist.all_gather_into_tensor(flat_param, shard)
                param.copy_(flat_param[: param.numel()].view_as(param))

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
