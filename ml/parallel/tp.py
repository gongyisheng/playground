"""Megatron-style tensor parallel dense MLP.

Y = GeLU(X @ W1) @ W2, with W1 column-parallel and W2 row-parallel so the
whole MLP needs just ONE all-reduce in forward (and one in backward).

Run with: uv run torchrun --nproc_per_node=2 tp.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# x: [M, K]
# fc1: [K, N] ---> lets do column wise ---> [K, N/2], [K, N/2]
# x@fc1_0, x@fc1_1 ---> [M, N/2], [M, N/2]
# fc2: [N, O] ---> lets do row wise ---> [N/2, O], [N/2, O]
# x_0@fc2_0, x1@fc2_1 ---> [M, O], [M, O] ---> need all_reduce to get final [M, O]


class _ReduceBackward(torch.autograd.Function):
    """f: identity forward, all-reduce backward. Used at column-parallel input."""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad)
        return grad


class _ReduceForward(torch.autograd.Function):
    """g: all-reduce forward, identity backward. Used at row-parallel output."""
    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


class ColumnParallelLinear(nn.Module):
    """Shard weight on out_features. Input replicated, output sharded."""
    def __init__(self, in_f, out_f, world):
        super().__init__()
        assert out_f % world == 0
        self.w = nn.Parameter(torch.empty(out_f // world, in_f))
        self.b = nn.Parameter(torch.empty(out_f // world))

    def forward(self, x):
        x = _ReduceBackward.apply(x)
        return F.linear(x, self.w, self.b)


class RowParallelLinear(nn.Module):
    """Shard weight on in_features. Input sharded, output reduced."""
    def __init__(self, in_f, out_f, world):
        super().__init__()
        assert in_f % world == 0
        self.w = nn.Parameter(torch.empty(out_f, in_f // world))
        self.b = nn.Parameter(torch.empty(out_f))

    def forward(self, x):
        y = F.linear(x, self.w)          # partial [b, out_f], no bias yet
        y = _ReduceForward.apply(y)      # the one all-reduce
        return y + self.b                # full bias added once per rank


class DenseMLP(nn.Module):
    def __init__(self, hidden, world):
        super().__init__()
        self.fc1 = ColumnParallelLinear(hidden, 4 * hidden, world)
        self.fc2 = RowParallelLinear(4 * hidden, hidden, world)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def load_sharded(mlp, w1, b1, w2, b2, rank, world):
    """Slice full reference weights into this rank's shards.

    nn.Linear weight is [out, in]. Column-parallel splits out (dim 0);
    row-parallel splits in (dim 1). Bias: fc1 sharded, fc2 full.
    """
    per = w1.shape[0] // world
    sl = slice(rank * per, (rank + 1) * per)
    with torch.no_grad():
        mlp.fc1.w.copy_(w1[sl])
        mlp.fc1.b.copy_(b1[sl])
        mlp.fc2.w.copy_(w2[:, sl])
        mlp.fc2.b.copy_(b2)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    b, hidden = 8, 512
    torch.manual_seed(0)  # identical full weights + input on every rank

    # full reference weights (replicated on each rank)
    w1 = torch.randn(4 * hidden, hidden, device=dev) * 0.02
    b1 = torch.randn(4 * hidden, device=dev) * 0.02
    w2 = torch.randn(hidden, 4 * hidden, device=dev) * 0.02
    b2 = torch.randn(hidden, device=dev) * 0.02
    x = torch.randn(b, hidden, device=dev)

    # reference: full MLP, replicated
    x_ref = x.clone().requires_grad_(True)
    y_ref = F.linear(F.gelu(F.linear(x_ref, w1, b1)), w2, b2)
    y_ref.sum().backward()

    # tensor-parallel MLP
    mlp = DenseMLP(hidden, world).to(dev)
    load_sharded(mlp, w1, b1, w2, b2, rank, world)
    x_tp = x.clone().requires_grad_(True)
    y_tp = mlp(x_tp)
    y_tp.sum().backward()

    if rank == 0:
        fwd_ok = torch.allclose(y_tp, y_ref, rtol=1e-3, atol=1e-4)
        bwd_ok = torch.allclose(x_tp.grad, x_ref.grad, rtol=1e-3, atol=1e-4)
        print(f"world={world}  forward match: {fwd_ok}  input-grad match: {bwd_ok}")
        print(f"  max fwd err: {(y_tp - y_ref).abs().max():.2e}")
        print(f"  max dx  err: {(x_tp.grad - x_ref.grad).abs().max():.2e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
