"""Minimal GPTQ: int4 weight quant with Hessian-based error compensation.

GPTQ quantizes a weight matrix one input-column at a time; after rounding each
column it pushes the resulting error into the not-yet-quantized columns using the
layer's Hessian H = XᵀX from calibration data, so the layer output X·Wᵀ barely
moves. Compared here against plain round-to-nearest (RTN) at the same int4 scale.

Toy linear layer: W is (out, in), calibration X is (n, in), output Y = X·Wᵀ.
"""

import torch

QMIN, QMAX = -8, 7  # int4


def quantize(W: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Symmetric int4, per-row (per output channel) scale; returns dequantized W."""
    q = torch.clamp(torch.round(W / scale), QMIN, QMAX)
    return q * scale


def gptq(W: torch.Tensor, X: torch.Tensor, scale: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    W = W.clone().float()
    n_in = W.shape[1]
    s = scale[:, 0]  # per-row scale as (out,)

    H = 2 * X.T @ X  # (in, in) layer Hessian from calibration activations
    idx = torch.arange(n_in)
    H[idx, idx] += damp * torch.diag(H).mean()  # dampen for a stable inverse
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))

    for j in range(n_in):
        w = W[:, j]
        q = torch.clamp(torch.round(w / s), QMIN, QMAX) * s
        err = (w - q) / Hinv[j, j]  # error scaled by output-sensitivity of this column
        W[:, j] = q
        if j + 1 < n_in:
            # push the error into the remaining columns, weighted by their coupling to j
            W[:, j + 1:] -= err[:, None] * Hinv[j, j + 1:][None, :]
    return W


if __name__ == "__main__":
    torch.manual_seed(0)
    out_f, in_f, n = 256, 512, 512
    W = torch.randn(out_f, in_f)

    X = torch.randn(n, in_f)
    X += torch.randn(n, 1) * 2  # shared component -> correlated columns, so H has off-diagonals
    X[:, :16] *= 5              # a few salient (high-magnitude) input channels

    scale = W.abs().max(dim=1, keepdim=True).values / QMAX  # per output channel
    Y = X @ W.T

    for name, Wq in [("rtn", quantize(W, scale)), ("gptq", gptq(W, X, scale))]:
        Yq = X @ Wq.T
        weight_mse = torch.mean((W - Wq) ** 2).item()
        output_mse = torch.mean((Y - Yq) ** 2).item()
        print(f"[{name:>4}] weight_MSE={weight_mse:.3e} output_MSE={output_mse:.3e}")
