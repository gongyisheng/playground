"""Asymmetric per-tensor int8 quantization (bf16 / fp16 / fp32 -> int8)."""

import torch

QMIN, QMAX = -128, 127  # int8 range


def compute_qparams(x: torch.Tensor) -> tuple[float, int]:
    x_min = x.min().item()
    x_max = x.max().item()
    scale = (x_max - x_min) / (QMAX - QMIN)
    if scale == 0:
        scale = 1.0
    zero_point = round(QMIN - x_min / scale)
    zero_point = int(max(QMIN, min(QMAX, zero_point)))
    return scale, zero_point


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    x = x.to(torch.float32)  # upcast so scale math isn't polluted by bf16/fp16 rounding
    scale, zero_point = compute_qparams(x)
    q = torch.clamp(torch.round(x / scale) + zero_point, QMIN, QMAX)
    return q.to(torch.int8), scale, zero_point


def dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (q.to(torch.float32) - zero_point) * scale


def report(name: str, x: torch.Tensor) -> None:
    q, scale, zero_point = quantize(x)
    x_hat = dequantize(q, scale, zero_point)
    ref = x.to(torch.float32)
    mse = torch.mean((ref - x_hat) ** 2).item()
    max_err = torch.max(torch.abs(ref - x_hat)).item()
    print(f"[{name:>4}] scale={scale:.6g} zero_point={zero_point:4d} "
          f"MSE={mse:.3e} max_err={max_err:.3e}")


if __name__ == "__main__":
    torch.manual_seed(42)
    base = torch.randn(256, 256)
    for name, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)]:
        report(name, base.to(dtype))
