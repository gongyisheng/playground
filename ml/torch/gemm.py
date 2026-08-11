import torch

# ---------------------------------------------------------------------------
# 1. basic gemm
# ---------------------------------------------------------------------------

def test_mm():
    a = torch.randn(4, 8)
    b = torch.randn(8, 16)

    # @ / torch.matmul / torch.mm all do the same 2d gemm
    print(torch.equal(a @ b, torch.matmul(a, b)))  # True
    print(torch.equal(a @ b, torch.mm(a, b)))      # True

    # mm is strict 2d only, no broadcast and no dim promotion
    try:
        torch.mm(torch.randn(2, 4, 8), b)
    except RuntimeError as e:
        print("mm 3d:", e)  # self must be a matrix

    # out= writes into a preallocated buffer, no new allocation
    out = torch.empty(4, 16)
    print(torch.mm(a, b, out=out).data_ptr() == out.data_ptr())  # True

def test_bmm():
    a = torch.randn(2, 4, 8)
    b = torch.randn(2, 8, 16)

    # bmm: strict 3d x 3d, batch sizes must match exactly (no broadcast)
    print(torch.allclose(torch.bmm(a, b), a @ b, atol=1e-5))  # True
    try:
        torch.bmm(torch.randn(1, 4, 8), b)
    except RuntimeError as e:
        print("bmm broadcast:", e)  # expects [1,8] for batch2, got [2,8]
    try:
        torch.bmm(a, torch.randn(8, 16))
    except RuntimeError as e:
        print("bmm 2d:", e)  # batch2 must be a 3D tensor

    # matmul does broadcast the batch dim, so it accepts what bmm rejects
    print(torch.matmul(torch.randn(1, 4, 8), b).shape)  # [2,4,16]
    # ...and a bare 2d operand is treated as shared across the batch
    print(torch.matmul(a, torch.randn(8, 16)).shape)  # [2,4,16]

    # >3d: matmul flattens all leading dims into one batch and calls bmm

def test_addmm():
    a = torch.randn(4, 8)
    b = torch.randn(8, 16)
    bias = torch.randn(16)

    # addmm fuses bias into the epilogue: beta*bias + alpha*(a@b).
    # this is what nn.Linear dispatches to, so the add costs nothing extra.
    print(torch.allclose(torch.addmm(bias, a, b), bias + a @ b, atol=1e-6))  # True
    print(torch.allclose(torch.addmm(bias, a, b, beta=2.0, alpha=0.5),
                         2.0 * bias + 0.5 * (a @ b), atol=1e-6))  # True

    # bias broadcasts over rows, so a (4,16) or (1,16) bias works too
    print(torch.addmm(torch.randn(4, 16), a, b).shape)  # [4,16]

def test_matmul_broadcast():
    # matmul rules by ndim:
    # 1d x 1d -> dot product (scalar)
    print(torch.matmul(torch.randn(8), torch.randn(8)).shape)  # []

    # 1d x 2d -> vector is prepended a 1 then removed: (8) @ (8,16) -> (16)
    print(torch.matmul(torch.randn(8), torch.randn(8, 16)).shape)  # [16]

    # 2d x 1d -> vector is appended a 1 then removed: (4,8) @ (8) -> (4)
    print(torch.matmul(torch.randn(4, 8), torch.randn(8)).shape)  # [4]

    # nd: last 2 dims are the matrix, leading dims broadcast
    x = torch.randn(2, 1, 4, 8)
    y = torch.randn(3, 8, 16)
    print(torch.matmul(x, y).shape)  # [2,3,4,16]

    # broadcast is a view, no copy of y across the batch dim -> cheap

def test_matmul_dtype():
    a = torch.randn(4, 8, dtype=torch.float32)
    b = torch.randn(8, 16, dtype=torch.bfloat16)

    # no implicit dtype promotion in matmul, unlike elementwise ops
    try:
        a @ b
    except RuntimeError as e:
        print("mixed dtype:", e)

    # accumulation is always in fp32 for bf16/fp16 inputs, output is bf16
    ab = a.bfloat16() @ b
    print(ab.dtype)  # torch.bfloat16

    # float32 matmul precision: "highest"=true fp32, "high"=tf32, "medium"=bf16
    # tf32 keeps 10 mantissa bits -> ~1e-3 relative error, ~8x faster on tensor
    # cores. cuda only: on cpu all three modes give bit-identical results.
    for mode in ("highest", "high"):
        torch.set_float32_matmul_precision(mode)
        print(mode, (a @ b.float()).abs().mean().item())
    torch.set_float32_matmul_precision("highest")


# ---------------------------------------------------------------------------
# 2. torch._grouped_mm
#
# aten::_grouped_mm(self, mat2, offs=None, bias=None, out_dtype=None)
#
# runs G independent gemms with different shapes in one kernel launch. built
# for MoE: each expert is a gemm over a variable number of routed tokens.
# offs is int32, cumulative *end* offsets (exclusive prefix sum shifted by 1),
# and which dim it partitions depends on the ndim of the two operands.
#
# constraints: all strides must be a multiple of 16 bytes (so for bf16 the
# leading dim must be a multiple of 8), and offs must be int32. the fast cuda
# path (sm90+ grouped cutlass kernel) wants bf16; the cpu fallback used here
# also accepts fp32, so don't rely on dtype checks passing locally.
# ---------------------------------------------------------------------------

def test_grouped_mm_2d3d():
    # 2d x 3d: offs splits A's rows (M) into groups, group g uses B[g].
    # this is the MoE forward: tokens sorted by expert, one weight per expert.
    a = torch.randn(8, 16, dtype=torch.bfloat16)      # 8 tokens, K=16
    b = torch.randn(2, 16, 8, dtype=torch.bfloat16)   # 2 experts, K=16 -> N=8
    offs = torch.tensor([3, 8], dtype=torch.int32)    # rows 0:3 -> e0, 3:8 -> e1

    out = torch._grouped_mm(a, b, offs=offs)
    print(out.shape)  # [8,8] -> same M as A, no group dim
    print(torch.equal(out[:3], a[:3] @ b[0]), torch.equal(out[3:], a[3:] @ b[1]))  # True True

def test_grouped_mm_3d3d():
    # 3d x 3d: no offs, all groups have the same M -> this is just bmm
    a = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    b = torch.randn(2, 16, 8, dtype=torch.bfloat16)
    out = torch._grouped_mm(a, b)
    print(out.shape, torch.equal(out, torch.bmm(a, b)))  # [2,8,8] True

def test_grouped_mm_3d2d():
    # 3d x 2d: offs splits B's columns (N); group g is A[g] @ B[:, start:end]
    a = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    b = torch.randn(16, 8, dtype=torch.bfloat16)
    offs = torch.tensor([3, 8], dtype=torch.int32)

    out = torch._grouped_mm(a, b, offs=offs)
    print(out.shape)  # [8,8] -> N is the concatenation of the group slices
    print(torch.equal(out[:, :3], a[0] @ b[:, :3]), torch.equal(out[:, 3:], a[1] @ b[:, 3:]))

def test_grouped_mm_2d2d():
    # 2d x 2d: neither operand has a group axis, so the only thing that can
    # vary per group is the reduction extent -> offs splits K, and the group
    # dim appears in the OUTPUT. each group is a full (M,N) matrix.
    a = torch.randn(8, 16, dtype=torch.bfloat16)
    b = torch.randn(16, 8, dtype=torch.bfloat16)
    offs = torch.tensor([8, 16], dtype=torch.int32)  # K 0:8 -> g0, 8:16 -> g1

    out = torch._grouped_mm(a, b, offs=offs)
    print(out.shape)  # [2,8,8] -> (G,M,N)
    print(torch.equal(out[0], a[:, :8] @ b[:8]), torch.equal(out[1], a[:, 8:] @ b[8:]))
    # here the groups happen to tile all of K, so they sum to the full gemm.
    # that is a property of this example, not the point of the op.
    print(torch.allclose(out.sum(0).float(), (a @ b).float(), atol=1e-1))

    # the real use is the MoE weight gradient, where the contraction dim is the
    # token dim and each expert owns a different token range:
    #   dW[e] = x[tokens of e].T @ dy[tokens of e]
    # so A = x.T is (K, T), B = dy is (T, N), offs splits T -> out is (E, K, N)
    x = torch.randn(8, 16)                            # 8 tokens, K=16
    dy = torch.randn(8, 8)                            # N=8
    tok = torch.tensor([2, 8], dtype=torch.int32)     # 2 tokens -> e0, 6 -> e1
    dw = torch._grouped_mm(x.t().contiguous(), dy, offs=tok)
    print(dw.shape)  # [2,16,8] -> one (K,N) gradient per expert
    print(torch.allclose(dw[0], x[:2].t() @ dy[:2], atol=1e-4),
          torch.allclose(dw[1], x[2:].t() @ dy[2:], atol=1e-4))  # True True

def test_grouped_mm_errors():
    a = torch.randn(8, 16, dtype=torch.bfloat16)
    b = torch.randn(2, 16, 8, dtype=torch.bfloat16)
    offs = torch.tensor([3, 8], dtype=torch.int32)

    # bias is in the schema but not implemented yet -> fold it in afterwards
    try:
        torch._grouped_mm(a, b, offs=offs, bias=torch.randn(2, 8, dtype=torch.bfloat16))
    except RuntimeError as e:
        print("bias:", e)  # Bias not supported yet

    # out_dtype is in the schema but must equal the input dtype -> also a no-op
    try:
        torch._grouped_mm(a, b, offs=offs, out_dtype=torch.float32)
    except RuntimeError as e:
        print("out_dtype:", e)

    # offs must be int32, not int64
    try:
        torch._grouped_mm(a, b, offs=offs.long())
    except RuntimeError as e:
        print("offs dtype:", e)

    # 16-byte stride alignment: K=15 bf16 rows are 30 bytes -> rejected
    try:
        torch._grouped_mm(torch.randn(8, 15, dtype=torch.bfloat16),
                          torch.randn(2, 15, 8, dtype=torch.bfloat16), offs=offs)
    except RuntimeError as e:
        print("align:", e)

    # empty groups are legal: repeating an offset gives a zero-row group
    print(torch._grouped_mm(a, b, offs=torch.tensor([0, 8], dtype=torch.int32)).shape)  # [8,8]


if __name__ == "__main__":
    test_mm()
    # test_bmm()
    # test_addmm()
    # test_matmul_broadcast()
    # test_matmul_dtype()
    # test_grouped_mm_2d3d()
    # test_grouped_mm_3d3d()
    # test_grouped_mm_3d2d()
    # test_grouped_mm_2d2d()
    # test_grouped_mm_errors()
