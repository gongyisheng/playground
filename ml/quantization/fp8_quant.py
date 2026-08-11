"""Symmetric per-tensor fp8 quantization (bf16 / fp16 / fp32 -> E4M3 / E5M2).

fp8 is a non-uniform grid (sign/exponent/mantissa), so unlike int8 there is no
zero_point: fp8 represents 0 exactly and is symmetric. We only pick a scale that
maps the tensor's largest magnitude onto the format's max representable value
(absmax scaling), then let the hardware cast round to the nearest fp8 code.
"""

import torch


def compute_scale(x: torch.Tensor, fp8_dtype: torch.dtype) -> float:
    fp8_max = torch.finfo(fp8_dtype).max
    scale = x.abs().max().item() / fp8_max
    if scale == 0:
        scale = 1.0
    return scale


def quantize(x: torch.Tensor, fp8_dtype: torch.dtype) -> tuple[torch.Tensor, float]:
    x = x.to(torch.float32)  # upcast so scale math isn't polluted by bf16/fp16 rounding
    scale = compute_scale(x, fp8_dtype)
    q = (x / scale).to(fp8_dtype)  # round-to-nearest-even + saturate, done by the cast
    return q, scale


def dequantize(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.to(torch.float32) * scale


def report(name: str, x: torch.Tensor, fp8_dtype: torch.dtype) -> None:
    q, scale = quantize(x, fp8_dtype)
    x_hat = dequantize(q, scale)
    ref = x.to(torch.float32)
    mse = torch.mean((ref - x_hat) ** 2).item()
    max_err = torch.max(torch.abs(ref - x_hat)).item()
    print(f"[{name:>4}] scale={scale:.6g} "
          f"MSE={mse:.3e} max_err={max_err:.3e}")


if __name__ == "__main__":
    torch.manual_seed(42)
    base = torch.randn(256, 256)
    for name, fp8_dtype in [("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)]:
        report(name, base, fp8_dtype)
