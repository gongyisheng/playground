"""Sweep int8 per-tensor quant error to find where it breaks vs stays safe.

Reuses the quantizer from int8_quant.py. Goal: show error as a function of the
distribution's dynamic range (driven here by outliers), and confirm that the
single predictor SNR ≈ 12 · 255² · (std/range)² tracks measured error.
"""

import torch

from int8_quant import quantize, dequantize


def measure(x: torch.Tensor) -> dict:
    """Quantize x, dequantize, return error + predictor stats for one tensor."""
    x = x.to(torch.float32)
    q, scale, zero_point = quantize(x)
    x_hat = dequantize(q, scale, zero_point)

    mse = torch.mean((x - x_hat) ** 2).item()
    signal_power = torch.var(x).item()
    snr_db = 10 * torch.log10(torch.tensor(signal_power / mse)).item()

    std = x.std().item()
    rng = (x.max() - x.min()).item()
    return {
        "scale": scale,
        "mse": mse,
        "snr_db": snr_db,
        "std_over_range": std / rng,          # the theoretical predictor
        "used_levels": q.unique().numel(),     # distinct int8 codes actually used, of 256
    }


def make_tensor(outlier_scale: float, n: int = 256 * 256, frac: float = 0.001) -> torch.Tensor:
    """Gaussian bulk (std 1) with 0.1% one-sided outliers at `outlier_scale`·std.

    One-sided spikes are the harsh, realistic case: they stretch the range AND
    drag zero_point off-center, exactly what wrecks per-tensor quant on LLM
    activations. outlier_scale == 0 returns the clean "safe" baseline.
    """
    x = torch.randn(n)
    if outlier_scale > 0:
        k = max(1, int(n * frac))
        idx = torch.randperm(n)[:k]
        x[idx] = outlier_scale  # units of std, one-sided (+)
    return x


if __name__ == "__main__":
    torch.manual_seed(42)
    print(f"{'outlier×std':>11} {'scale':>10} {'used/256':>9} "
          f"{'std/range':>10} {'MSE':>10} {'SNR(dB)':>9}")
    for outlier_scale in [0, 5, 10, 20, 50, 100]:
        x = make_tensor(outlier_scale)
        m = measure(x)
        print(f"{outlier_scale:>11} {m['scale']:>10.4g} {m['used_levels']:>9d} "
              f"{m['std_over_range']:>10.4f} {m['mse']:>10.3e} {m['snr_db']:>9.2f}")
