import math

import torch


def hadamard_matrix(n: int) -> torch.Tensor:
    """Sylvester construction"""
    assert n & (n-1) == 0, f"Sylvester construction needs a power of 2, got {n}"
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n**0.5


def fwht(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    n = shape[-1]
    assert n & (n - 1) == 0, f"fwht needs a power-of-2 last dim, got {n}"
    h = 1
    while h < n:
        x = x.reshape(-1, n//(2*h), 2, h)
        a, b = x[:, :, 0], x[:, :, 1]
        x = torch.stack([a+b, a-b], dim=2)
        h *= 2
    return x.reshape(shape) / n**0.5


def incoherence(x: torch.Tensor) -> torch.Tensor:
    """sqrt(n) * ||x||_inf / ||x||_2, per row.

    1.0 = energy spread perfectly flat, sqrt(n) = all energy on one spike.
    A Hadamard rotation should pull this toward sqrt(2 ln n)."""
    n = x.shape[-1]
    return n**0.5 * x.abs().amax(-1) / x.norm(dim=-1)


def random_signs(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=g).float() * 2 - 1


def rotate_linear(w: torch.Tensor, signs: torch.Tensor):
    """Fold the rotation into a linear layer's weight, offline.

    For y = x @ w, insert the identity H H^T:  y = (x @ H) @ (H^T @ w).
    The right factor is precomputed here and is what you ship and quantize;
    at runtime the activation only pays one fwht. Returns (w_rot, apply_fn)
    where apply_fn does the runtime-side transform of x.
    """
    # Runtime side is x -> (x D) H, so the weight side must be
    # (D H)^-1 w = H D w. Both D and H are self-inverse, so only the order
    # flips. fwht hits the last dim, hence the transpose to reach w's rows
    # (valid because Sylvester's H is symmetric).
    w_rot = fwht((signs[:, None] * w).T).T
    return w_rot, lambda x: fwht(x * signs)


def demo_outliers():
    torch.manual_seed(0)
    n = 4096
    x = torch.randn(8, n)
    x[:, 0] = 100.0  # one fat outlier per row, the classic LLM activation shape

    signs = random_signs(n)
    xr = fwht(x * signs)

    print(f"n={n}  sqrt(n)={n**0.5:.1f}  sqrt(2 ln n)={math.sqrt(2 * math.log(n)):.2f}")
    print(f"before: amax={x.abs().amax():7.2f}  mu={incoherence(x).mean():6.2f}")
    print(f"after : amax={xr.abs().amax():7.2f}  mu={incoherence(xr).mean():6.2f}")
    print(f"norm preserved: {torch.allclose(x.norm(dim=-1), xr.norm(dim=-1), atol=1e-3)}")


def test():
    n = 256
    h = hadamard_matrix(n)
    assert torch.allclose(h @ h.T, torch.eye(n), atol=1e-5), "H not orthogonal"

    x = torch.randn(4, n)
    assert torch.allclose(fwht(x), x @ h, atol=1e-4), "fwht != matmul with H"
    assert torch.allclose(fwht(fwht(x)), x, atol=1e-4), "H is its own inverse"

    w = torch.randn(n, 32)
    signs = random_signs(n)
    w_rot, apply_fn = rotate_linear(w, signs)
    assert torch.allclose(apply_fn(x) @ w_rot, x @ w, atol=1e-3), "rotation not invariant"
    print("all tests passed")


if __name__ == "__main__":
    test()
    demo_outliers()
