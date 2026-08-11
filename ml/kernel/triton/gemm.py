import torch
import triton
import triton.language as tl

@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)
    a_base_ptr = a_ptr + stride_am * offset_m[:, None] + stride_ak * offset_k[None, :]
    b_base_ptr = b_ptr + stride_bk * offset_k[:, None] + stride_bn * offset_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        mask_a = (offset_m[:, None] < M) & (offset_k[None, :] + k < K)
        frag_a = tl.load(a_base_ptr, mask=mask_a, other=0.0)
        mask_b = (offset_k[:, None] + k < K) & (offset_n[None, :] < N)
        frag_b = tl.load(b_base_ptr, mask=mask_b, other=0.0)
        acc += tl.dot(frag_a, frag_b)
        a_base_ptr += stride_ak * BLOCK_K
        b_base_ptr += stride_bk * BLOCK_K

    c_base_ptr = c_ptr + stride_cm * offset_m[:, None] + stride_cn * offset_n[None, :]
    mask_c = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    tl.store(c_base_ptr, acc.to(c_ptr.dtype.element_ty), mask=mask_c)


# a: [M, K], b: [K, N], c: [M, N]
# tile-based: each program store [m, n] -> BLOCK_M, BLOCK_N
# for a_i: [m, K], for b_i: [K, n], a_i @ b_i, -> BLOCK_K
def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # assert is_cuda
    # assert contiguous
    # assert dim == 2
    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2
    K = K1

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    c = torch.empty(M, N, dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return c


if __name__ == "__main__":
    a = torch.rand(333, 27, device="cuda", dtype=torch.bfloat16)
    b = torch.rand(27, 333, device="cuda", dtype=torch.bfloat16)
    c = gemm(a, b)
    c_ref = a @ b
    print(c)
    print(c_ref)
    print(torch.allclose(c, c_ref, rtol=1e-2, atol=1e-2))