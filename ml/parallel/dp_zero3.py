"""Minimal ZeRO Stage 3 data parallel training, using only PyTorch.

Run on two CPU workers:
    torchrun --standalone --nproc-per-node=2 dp_zero3.py

Run on two GPUs (one GPU per worker):
    torchrun --standalone --nproc-per-node=2 dp_zero3.py --device cuda

Each worker keeps only a slice of each parameter, averaged gradient, and AdamW
optimizer state. All-gather materializes a layer's weights for forward and again
for backward, then releases them. Custom autograd saves only inputs and local
shards, so full weights do not survive forward. Backward reduce-scatters each
gradient before returning its local slice. Tensors are padded for uneven splits.
This demo uses bias-free linear layers and synchronous collectives per tensor;
production implementations use buckets and overlap communication with computation.
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


def gather_weight(shard, shape):
    flat_weight = shard.new_empty(shard.numel() * dist.get_world_size())
    dist.all_gather_into_tensor(flat_weight, shard)
    return flat_weight[: shape.numel()].view(shape)


class _ShardedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shard, shape):
        ctx.save_for_backward(x, shard)
        ctx.shape = shape
        # Custom forward runs without recording a graph that would save weights.
        weight = gather_weight(shard, shape)
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, shard = ctx.saved_tensors
        world_size = dist.get_world_size()

        # Re-gather weights for the input gradient; release them before reducing.
        weight = gather_weight(shard, ctx.shape)
        grad_input = grad_output.matmul(weight) if ctx.needs_input_grad[0] else None
        del weight

        grad_weight = grad_output.reshape(-1, ctx.shape[0]).t().matmul(
            x.reshape(-1, ctx.shape[1])
        )
        flat_grad = F.pad(
            grad_weight.flatten(), (0, shard.numel() * world_size - ctx.shape.numel())
        )
        grad_shard = torch.empty_like(shard)
        # All workers run the same graph, so collectives use the same order.
        dist.reduce_scatter_tensor(grad_shard, flat_grad, op=dist.ReduceOp.SUM)
        # Equal local batch sizes make this the global mean-loss gradient slice.
        grad_shard.div_(world_size)
        return grad_input, grad_shard, None


class ShardedLinear(nn.Module):
    """Bias-free linear layer that stores only this worker's weight slice."""

    def __init__(self, in_features, out_features, device):
        super().__init__()
        rank, world_size = dist.get_rank(), dist.get_world_size()
        # Initialize as in the other demos, then discard the full weight tensor.
        weight = nn.Linear(in_features, out_features, bias=False).to(device).weight
        with torch.no_grad():
            dist.broadcast(weight, src=0)
        self.shape = weight.shape
        flat_weight = F.pad(
            weight.detach().flatten(), (0, -weight.numel() % world_size)
        )
        # Clone so the shard does not retain the full tensor's backing storage.
        self.shard = nn.Parameter(flat_weight.chunk(world_size)[rank].clone())

    def forward(self, x):
        return _ShardedLinear.apply(x, self.shard, self.shape)


def train(args, device):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(42)
    model = nn.Sequential(
        ShardedLinear(32, 32, device),
        nn.GELU(),
        ShardedLinear(32, 32, device),
        nn.GELU(),
        ShardedLinear(32, 32, device),
        nn.GELU(),
        ShardedLinear(32, 32, device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=0.01, foreach=False
    )
    x, y = make_batch(rank, args.batch_size, device)

    if rank == 0:
        print(f"ZeRO-3 demo: {world_size} workers, device={device.type}", flush=True)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        # 1. Gather each layer's weights for forward, releasing them after use.
        loss = F.mse_loss(model(x), y)

        # 2. Re-gather for backward and reduce-scatter gradients into local slices.
        loss.backward()

        # 3. Update only local slices; full weights wait until the next forward.
        optimizer.step()

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
