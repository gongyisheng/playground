"""Symmetric per-tensor fp4 quantization (bf16 / fp16 / fp32 -> E2M1).

fp4 (E2M1: 1 sign, 2 exponent, 1 mantissa) has only 16 codes / 8 magnitudes, a
non-uniform grid with no zero_point. There is no eager fp4 cast in torch
(torch.float4_e2m1fn_x2 is a packed 2-per-byte storage dtype), so we simulate the
grid: pick an absmax scale, then round to the nearest representable level.
"""

import torch

# The 8 representable magnitudes of E2M1; full grid is these ± sign. Max = 6.0.
FP4_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX = 6.0


def compute_scale(x: torch.Tensor) -> float:
    scale = x.abs().max().item() / FP4_MAX
    if scale == 0:
        scale = 1.0
    return scale


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    x = x.to(torch.float32)  # upcast so scale math isn't polluted by bf16/fp16 rounding
    scale = compute_scale(x)
    x_scaled = x / scale

    # nearest-neighbour rounding onto the non-uniform grid: bucketize against the
    # midpoints between adjacent levels. Values past the last midpoint saturate to 6.0.
    levels = FP4_E2M1_LEVELS.to(x.device)
    midpoints = (levels[1:] + levels[:-1]) / 2
    idx = torch.bucketize(x_scaled.abs(), midpoints)
    q = torch.sign(x_scaled) * levels[idx]  # grid values, stored as fp32
    return q, scale


def dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q * scale


def report(name: str, x: torch.Tensor) -> None:
    q, scale = quantize(x)
    x_hat = dequantize(q, scale)
    ref = x.to(torch.float32)
    mse = torch.mean((ref - x_hat) ** 2).item()
    max_err = torch.max(torch.abs(ref - x_hat)).item()
    print(f"[{name:>4}] scale={scale:.6g} "
          f"MSE={mse:.3e} max_err={max_err:.3e}")


if __name__ == "__main__":
    torch.manual_seed(42)
    base = torch.randn(256, 256)
    for name, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)]:
        report(name, base.to(dtype))
