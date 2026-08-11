"""Minimal AWQ: activation-aware int4 weight quant via per-channel scaling.

AWQ protects the weight channels that multiply large activations. For a positive
per-input-channel scale s, the layer is unchanged — (X/s)·(s·W)ᵀ == X·Wᵀ — but
scaling the salient columns UP before rounding shrinks their relative int4 error.
s is chosen as act_magnitude**alpha, with alpha grid-searched to minimize output
error. Compared against plain round-to-nearest (RTN), which is exactly alpha=0.

Toy linear layer: W is (out, in), calibration X is (n, in), output Y = X·Wᵀ.
"""

import torch

QMIN, QMAX = -8, 7  # int4


def quantize(W: torch.Tensor, X: torch.Tensor, alpha: float) -> torch.Tensor:
    """Scale salient channels by act**alpha, int4-quantize, scale back. Returns dequant W."""
    act = X.abs().mean(dim=0)          # per-input-channel activation magnitude (in,)
    s = act.pow(alpha)
    s = s / s.mean()                   # normalize so overall weight magnitude is preserved

    Ws = W * s[None, :]                # scale columns up before rounding
    scale = Ws.abs().max(dim=1, keepdim=True).values / QMAX
    Wq = torch.clamp(torch.round(Ws / scale), QMIN, QMAX) * scale
    return Wq / s[None, :]             # fold the scale back -> effective quantized W


if __name__ == "__main__":
    torch.manual_seed(0)
    out_f, in_f, n = 256, 512, 512
    W = torch.randn(out_f, in_f)

    X = torch.randn(n, in_f)
    X[:, :16] *= 10  # 16 salient (outlier) input channels AWQ should protect

    Y = X @ W.T
    print(f"{'alpha':>6} {'output_MSE':>12}")
    for alpha in torch.linspace(0, 1, 11):
        Wq = quantize(W, X, alpha.item())
        output_mse = torch.mean((Y - X @ Wq.T) ** 2).item()
        tag = "  <- RTN" if alpha == 0 else ""
        print(f"{alpha.item():>6.1f} {output_mse:>12.3e}{tag}")
