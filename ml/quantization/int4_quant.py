import torch

QMIN, QMAX = -8, 7

# [v_min, v_max] -> [q_min, q_max]
# q = v/scale + zero_point
# v = (q - zero_point) * scale
# scale = (v_max-v_min)/(q_max-q_min) (float)
# zero_point = q - v/scale (int)


def compute_qparams(x: torch.Tensor) -> tuple[float, int]:
    x_max = x.max().item()
    x_min = x.min().item()
    scale = (x_max - x_min) / (QMAX - QMIN)
    if scale == 0:
        scale = 1.0
    zero_point = round(QMIN - x_min / scale)
    zero_point = int(max(QMIN, min(QMAX, zero_point)))
    return scale, zero_point


def quantize(x: torch.Tensor):
    x = x.to(torch.float32)
    scale, zero_point = compute_qparams(x)
    q = torch.clamp(torch.round(x/scale)+ zero_point, QMIN, QMAX)
    return q, scale, zero_point


def dequantize(x: torch.Tensor, scale: float, zero_point: int):
    return (x.to(torch.float32) - zero_point) * scale


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
    base = torch.randn(256, 256) * 2 + 10  # off-center, mimics a real activation/weight tensor
    for name, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)]:
        report(name, base.to(dtype))