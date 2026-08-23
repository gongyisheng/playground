import torch

# name -> (max_repr, storage dtype, scale kind, mantissa bits, smallest step)
# mantissa bits / smallest step describe the code grid, needed by stochastic
# rounding; mantissa bits None means "uniform grid of step 1" (int8).
FMT = {
    "int8": (127.0, torch.int8, "fp32", None, 1.0),
    "fp8_e4m3": (torch.finfo(torch.float8_e4m3fn).max, torch.float8_e4m3fn, "fp32", 3, 2**-9),
    "fp8_e5m2": (torch.finfo(torch.float8_e5m2).max, torch.float8_e5m2, "fp32", 2, 2**-16),
    "mxfp8": (torch.finfo(torch.float8_e4m3fn).max, torch.float8_e4m3fn, "e8m0", 3, 2**-9),
}

MX_BLOCK_SHAPE = (1, 32)  # fixed by the OCP microscaling spec


def _to_blocks(x: torch.Tensor, block_shape: tuple[int, int]) -> torch.Tensor:
    """Reshape a 2D tensor to (n_row_blocks, br, n_col_blocks, bc), zero-padding
    the tail. -1 in block_shape means "the whole axis"."""
    rows, cols = x.shape
    br = rows if block_shape[0] == -1 else block_shape[0]
    bc = cols if block_shape[1] == -1 else block_shape[1]
    x = torch.nn.functional.pad(x, (0, -cols % bc, 0, -rows % br))
    return x.reshape(x.shape[0] // br, br, x.shape[1] // bc, bc)


def _grid_step(a: torch.Tensor, mant_bits: int | None, min_step: float) -> torch.Tensor:
    """Distance between adjacent representable codes at magnitude `a`.

    Inside a binade [2^e, 2^(e+1)) a float grid has step 2^(e - mant_bits);
    below the smallest normal that flattens out to a fixed subnormal step."""
    if mant_bits is None:
        return torch.full_like(a, min_step)
    return torch.exp2(torch.floor(torch.log2(a)) - mant_bits).clamp_min(min_step)


def _stochastic_round(v: torch.Tensor, mant_bits: int | None, min_step: float):
    """Round to the grid below with prob 1-p, above with prob p, where p is the
    fractional position between them. Unbiased: E[q] == v."""
    a = v.abs()
    step = _grid_step(a, mant_bits, min_step)
    lo = torch.floor(a / step) * step
    up = torch.rand_like(a) < (a - lo) / step
    return torch.where(v < 0, -1.0, 1.0) * (lo + step * up)


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform over the last dim (must be a power of 2),
    scaled by 1/sqrt(n) so the transform is orthonormal."""
    shape = x.shape
    n = shape[-1]
    assert n & (n - 1) == 0, f"Hadamard needs a power-of-2 last dim, got {n}"
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a, b = x[:, :, 0], x[:, :, 1]
        x = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return x.reshape(shape) / n**0.5


def rht(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Randomized Hadamard transform: H @ (D x) with D a random +-1 diagonal.
    Each output is a signed average of every input in the row, so outliers get
    smeared across the row and the per-block amax drops."""
    return _fwht(x * signs)


def irht(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Inverse of rht. H and D are both orthogonal and self-inverse."""
    return _fwht(x) * signs


def random_signs(n: int, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2, (n,), generator=g).float() * 2 - 1


def quantize(x: torch.Tensor, fmt: str, block_shape: tuple[int, int], rounding: str = "rtn"):
    """Quantize a 2D tensor. Returns (blocked codes, blocked scales).

    rounding: "rtn" (round-to-nearest) or "sr" (stochastic)."""
    max_repr, dtype, scale_kind, mant_bits, min_step = FMT[fmt]
    if scale_kind == "e8m0":
        block_shape = MX_BLOCK_SHAPE

    xb = _to_blocks(x.float(), block_shape)
    amax = xb.abs().amax(dim=(1, 3), keepdim=True)
    # all-zero block would give scale 0; 1.0 leaves it exactly representable
    scale = torch.where(amax > 0, amax / max_repr, torch.ones_like(amax))
    if scale_kind == "e8m0":
        scale = torch.exp2(torch.ceil(torch.log2(scale)))  # ceil => never clips

    v = xb / scale
    if rounding == "sr":
        # already on the code grid, so the cast below is lossless
        q = _stochastic_round(v, mant_bits, min_step)
    elif dtype == torch.int8:
        q = v.round()  # the int cast truncates, fp8 casts round-to-nearest
    else:
        q = v
    return q.clamp(-max_repr, max_repr).to(dtype), scale


def dequantize(xq: torch.Tensor, scale: torch.Tensor, shape: tuple[int, int]):
    """Undo quantize: scale back up, unblock, drop the padding."""
    x = xq.float() * scale
    rows, cols = shape
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
    return x[:rows, :cols]


def sqnr(x: torch.Tensor, xhat: torch.Tensor) -> float:
    signal = x.float().pow(2).sum()
    noise = (x.float() - xhat).pow(2).sum()
    if noise == 0:
        return float("inf")
    return (10 * torch.log10(signal / noise)).item()


def _fmt_sqnr(
    x: torch.Tensor,
    fmt: str,
    block_shape: tuple[int, int],
    rounding: str = "rtn",
    signs: torch.Tensor | None = None,
) -> float:
    """Round-trip SQNR. With `signs`, quantization happens in the rotated domain
    and the result is rotated back before comparing against the original x."""
    xt = rht(x, signs) if signs is not None else x
    xq, scale = quantize(xt, fmt, block_shape, rounding)
    xhat = dequantize(xq, scale, xt.shape)
    return sqnr(x, irht(xhat, signs) if signs is not None else xhat)


def int8_sqnr(x: torch.Tensor, block_shape: tuple[int, int] = (-1, -1)) -> float:
    return _fmt_sqnr(x, "int8", block_shape)


def fp8_e4m3_sqnr(x: torch.Tensor, block_shape: tuple[int, int] = (-1, -1)) -> float:
    return _fmt_sqnr(x, "fp8_e4m3", block_shape)


def fp8_e5m2_sqnr(x: torch.Tensor, block_shape: tuple[int, int] = (-1, -1)) -> float:
    return _fmt_sqnr(x, "fp8_e5m2", block_shape)


def mxfp8_sqnr(x: torch.Tensor, block_shape: tuple[int, int] = MX_BLOCK_SHAPE) -> float:
    """block_shape is ignored: mxfp8 always scales over 32 elements."""
    return _fmt_sqnr(x, "mxfp8", block_shape)


GRANULARITIES = {
    "per-tensor": (-1, -1),
    "per-row": (1, -1),
    **{f"block-1x{n}": (1, n) for n in (16, 32, 64, 128)},
    **{f"block-{n}x{n}": (n, n) for n in (16, 32, 64, 128)},
}


def sweep(name: str, x: torch.Tensor, rounding: str = "rtn", signs: torch.Tensor | None = None):
    tag = f"{rounding.upper()}{' + RHT' if signs is not None else ''}"
    print(f"\n=== {name}  [{tag}]  (shape {tuple(x.shape)}, amax {x.abs().max():.1f}) ===")
    print(f"{'granularity':<15}" + "".join(f"{fmt:>12}" for fmt in FMT))
    for gran, block_shape in GRANULARITIES.items():
        cells = []
        for fmt in FMT:
            # mxfp8's granularity is fixed, so only report it on its own row
            if FMT[fmt][2] == "e8m0" and block_shape != MX_BLOCK_SHAPE:
                cells.append(f"{'-':>12}")
            else:
                cells.append(f"{_fmt_sqnr(x, fmt, block_shape, rounding, signs):>12.2f}")
        print(f"{gran:<15}" + "".join(cells))


def ablate(cases, block_shape: tuple[int, int] = (1, 32)):
    """RHT x rounding, at one granularity, so the four arms sit side by side."""
    arms = [(r, s) for r in ("rtn", "sr") for s in (False, True)]
    print(f"\n=== ablation @ block {block_shape}: SQNR (dB) ===")
    header = "".join(f"{r.upper() + ('+RHT' if s else ''):>12}" for r, s in arms)
    print(f"{'case':<30}{'format':<12}{header}")
    for name, x in cases:
        signs = random_signs(x.shape[1])
        for fmt in FMT:
            bs = MX_BLOCK_SHAPE if FMT[fmt][2] == "e8m0" else block_shape
            cells = "".join(
                f"{_fmt_sqnr(x, fmt, bs, r, signs if s else None):>12.2f}" for r, s in arms
            )
            print(f"{name:<30}{fmt:<12}{cells}")


if __name__ == "__main__":
    torch.manual_seed(0)

    normal = torch.randn(512, 512)

    # 1% of entries blown up 100x - the classic activation-outlier problem
    outlier = normal.clone()
    mask = torch.rand_like(outlier) < 0.01
    outlier[mask] *= 100

    uniform = torch.rand(512, 512) * 2 - 1
    cases = [
        ("normal", normal),
        ("normal + 1% outliers (100x)", outlier),
        ("uniform [-1, 1]", uniform),
    ]

    for rounding in ("rtn", "sr"):
        for use_rht in (False, True):
            for name, x in cases:
                sweep(name, x, rounding, random_signs(x.shape[1]) if use_rht else None)

    ablate(cases)
