"""Sweep fp4 (E2M1) per-tensor quant error vs outliers, contrasted with int4.

Reuses the quantizer from fp4_quant.py. fp4 is a non-uniform (relative-error)
grid, so the uniform predictor SNR ≈ 12·L²·(std/range)² does NOT apply. But unlike
fp8, fp4's relative-error advantage does NOT save it from outliers: with only 8
magnitudes, one big outlier forces a large absmax scale and the whole bulk collapses
onto ~2 levels (watch used/15 fall as outlier×std grows). That collapse is why
per-tensor fp4 is unusable and block scaling (MXFP4/NVFP4) is mandatory, not optional.
"""

import torch

from fp4_quant import quantize, dequantize


def measure(x: torch.Tensor) -> dict:
    """Quantize x, dequantize, return error + distribution stats for one tensor."""
    x = x.to(torch.float32)
    q, scale = quantize(x)
    x_hat = dequantize(q, scale)

    mse = torch.mean((x - x_hat) ** 2).item()
    signal_power = torch.var(x).item()
    snr_db = 10 * torch.log10(torch.tensor(signal_power / mse)).item()

    std = x.std().item()
    rng = (x.max() - x.min()).item()
    return {
        "scale": scale,
        "mse": mse,
        "snr_db": snr_db,
        "std_over_range": std / rng,
        "used_levels": q.unique().numel(),  # distinct fp4 codes used, of 15
    }


def make_tensor(outlier_scale: float, n: int = 256 * 256, frac: float = 0.001) -> torch.Tensor:
    """Gaussian bulk (std 1) with 0.1% one-sided outliers at `outlier_scale`·std.

    One-sided spikes stretch the range, the harsh realistic case for per-tensor
    quant on LLM activations. outlier_scale == 0 returns the clean baseline.
    """
    x = torch.randn(n)
    if outlier_scale > 0:
        k = max(1, int(n * frac))
        idx = torch.randperm(n)[:k]
        x[idx] = outlier_scale  # units of std, one-sided (+)
    return x


if __name__ == "__main__":
    torch.manual_seed(42)
    print(f"{'outlier×std':>11} {'scale':>10} {'used/15':>9} "
          f"{'std/range':>10} {'MSE':>10} {'SNR(dB)':>9}")
    for outlier_scale in [0, 5, 10, 20, 50, 100]:
        x = make_tensor(outlier_scale)
        m = measure(x)
        print(f"{outlier_scale:>11} {m['scale']:>10.4g} {m['used_levels']:>9d} "
              f"{m['std_over_range']:>10.4f} {m['mse']:>10.3e} {m['snr_db']:>9.2f}")
