import torch


def test_transpose():
    q = torch.rand(1,4,3)
    print("Before transpose:")
    print(q.shape) # [1,4,3]
    print(q)

    q = q.transpose(-1,-2)
    print("After transpose:")
    print(q.shape) # [1,3,4]
    print(q)

def test_contiguous():
    # operations can break contiguity
    # transpose, permute, narrow, expand, view, tensor[::2]

    # need to be contiguous before
    # view/reshape, .numpy(), or other cuda operations

    q = torch.rand(2, 3, 4)
    print("Original tensor, is contiguous:", q.is_contiguous())

    q_transposed = q.transpose(1, 2)
    print("After transpose, is contiguous:", q_transposed.is_contiguous())

    q_contiguous = q_transposed.contiguous()
    print("After contiguous, is contiguous:", q_contiguous.is_contiguous())

def test_reshape():
    # reshape vs view: reshape always works, view requires contiguous memory
    x = torch.rand(2, 3, 4)
    print("original shape:", x.shape)  # [2, 3, 4]

    # reshape: flexible, may copy if non-contiguous
    y = x.reshape(6, 4)
    print("reshape(6,4):", y.shape)

    # view: zero-copy but fails on non-contiguous tensors
    z = x.view(2, 12)
    print("view(2,12):", z.shape)

    # -1 means "infer this dimension" from total elements and other dims
    # only one -1 allowed per reshape call
    w = x.reshape(-1, 4)  # 2*3*4 / 4 = 6 → [6, 4]
    print("reshape(-1,4):", w.shape)

    # common in attention: merge batch and heads, then restore
    batch, heads, seq, dim = 2, 4, 8, 16
    q = torch.rand(batch, heads, seq, dim)
    q = q.reshape(-1, seq, dim)   # [batch*heads, seq, dim] = [8, 8, 16]
    print("merged batch+heads:", q.shape)
    q = q.reshape(batch, heads, seq, dim)  # restore
    print("restored:", q.shape)

    # after transpose, view fails but reshape works
    t = x.transpose(1, 2)  # [2, 4, 3], non-contiguous
    print("transposed is contiguous:", t.is_contiguous())
    r = t.reshape(2, 12)   # works (may copy)
    print("reshape after transpose:", r.shape)
    # t.view(2, 12) would raise RuntimeError

def test_repeat_interleave():
    # repeat_interleave: repeat elements along a dim
    # used in GQA to expand KV heads to match Q heads
    x = torch.tensor([[1, 2], [3, 4]])
    print("original:", x)

    # repeat each row 3 times
    y = x.repeat_interleave(3, dim=0)
    print("repeat_interleave(3, dim=0):")
    print(y)  # [[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]]

    # repeat each column 2 times
    z = x.repeat_interleave(2, dim=1)
    print("repeat_interleave(2, dim=1):")
    print(z)  # [[1,1,2,2],[3,3,4,4]]

def test_masked_fill():
    # masked_fill_: set values to a constant where mask is True
    # used for causal attention masking (future tokens → -inf)
    scores = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]])
    mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()
    print("mask (True = future):")
    print(mask)

    scores.masked_fill_(mask, float('-inf'))
    print("after masked_fill_:")
    print(scores)
    # [[1, -inf, -inf],
    #  [4,  5,  -inf],
    #  [7,  8,    9 ]]

def test_scatter():
    # scatter_: place values at specific indices — used in top-k sampling
    # scatter_(dim, index, src): self[index[i]] = src[i]
    logits = torch.full((6,), -float('inf'))
    indices = torch.tensor([1, 4, 2])
    values = torch.tensor([5.0, 4.0, 3.0])

    logits.scatter_(0, indices, values)
    print("after scatter_:", logits)  # [-inf, 5.0, 3.0, -inf, 4.0, -inf]

def test_bool():
    # .bool(): cast tensor to boolean dtype
    x = torch.tensor([0, 1, 1, 0, 1])
    mask = x.bool()
    print("int tensor:", x)
    print("bool tensor:", mask)  # [False, True, True, False, True]

    # commonly used: float mask from triu → bool for masked_fill_
    float_mask = torch.triu(torch.ones(3, 3), diagonal=1)
    bool_mask = float_mask.bool()
    print("float mask dtype:", float_mask.dtype)
    print("bool mask dtype:", bool_mask.dtype)

def test_interleaved_slice():
    # ::2 and 1::2: split interleaved data into even/odd elements
    # used in SwiGLU where gate and up projections are fused as [g0,u0,g1,u1,...]
    x = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])

    gate = x[::2]    # start=0, step=2 → even indices
    up = x[1::2]     # start=1, step=2 → odd indices
    print("original:", x)
    print("x[::2] (even):", gate)   # [10, 30, 50, 70]
    print("x[1::2] (odd):", up)     # [20, 40, 60, 80]

    # with ... (ellipsis): slice last dim, keep all preceding dims
    y = torch.arange(24).reshape(2, 3, 4)
    print("y[..., ::2] shape:", y[..., ::2].shape)    # (2, 3, 2)
    print("y[..., 1::2] shape:", y[..., 1::2].shape)  # (2, 3, 2)

def test_unsqueeze():
    # unsqueeze: add a dimension of size 1 at the specified position
    # critical for correct broadcasting in element-wise multiply
    weights = torch.tensor([0.6, 0.7, 0.5])           # (3,)
    h = torch.randn(3, 8)                              # (3, 8)

    # without unsqueeze: (3,) broadcasts against dim=-1 (wrong!)
    # with unsqueeze(-1): (3, 1) * (3, 8) → (3, 8) (correct)
    weighted = weights.unsqueeze(-1) * h
    print("weights shape:", weights.shape)              # (3,)
    print("unsqueezed shape:", weights.unsqueeze(-1).shape)  # (3, 1)
    print("result shape:", weighted.shape)              # (3, 8)

    # unsqueeze(0) vs unsqueeze(-1)
    x = torch.tensor([1, 2, 3])
    print("unsqueeze(0):", x.unsqueeze(0).shape)   # (1, 3)
    print("unsqueeze(-1):", x.unsqueeze(-1).shape) # (3, 1)

def test_advanced_indexing():
    batch_size = 4
    log_probs_example = torch.tensor([
        [-2.1, -1.5, -0.8, -3.2, -2.7, -1.2],  # Sample 0
        [-1.3, -2.4, -0.5, -1.8, -2.1, -3.0],  # Sample 1
        [-0.9, -1.7, -2.3, -0.6, -1.1, -2.8],  # Sample 2
        [-2.5, -0.7, -1.9, -2.2, -1.4, -3.1],  # Sample 3
    ])
    targets_example = torch.tensor([2, 2, 3, 1])

    row_indices = torch.arange(batch_size)
    print(f"Row indices: {row_indices}")
    print(f"Column indices: {targets_example}")

    # advanced indexing
    target_log_probs_example = log_probs_example[row_indices, targets_example]
    print(f"Result: {target_log_probs_example}")

def test_fancy_indexing_shapes():
    # tensor index vs scalar index: tensor preserves dim, scalar removes it
    x = torch.randn(5, 3)

    # scalar index: removes dim 0
    row = x[0]
    print("x[0] shape:", row.shape)  # (3,)

    # tensor index: preserves dims, selects multiple rows
    idx = torch.tensor([0, 2, 4])
    rows = x[idx]
    print("x[tensor([0,2,4])] shape:", rows.shape)  # (3, 3)

    # 3D example: scalar removes the indexed dim
    y = torch.randn(8, 4, 16)
    print("y[2] shape:", y[2].shape)                    # (4, 16) — dim 0 removed
    print("y[tensor([2,5])] shape:", y[torch.tensor([2,5])].shape)  # (2, 4, 16) — dim 0 preserved

def test_pick_vector():
    # pick a vector along a dim with plain indexing
    # the fixed integer index marks which dim you slice; ':' keeps the other
    x = torch.tensor([[1, 2],
                      [3, 4]])

    # pick row 0: fix dim-0, keep all of dim-1
    row = x[0, :]
    print("x[0, :] row vector:", row, row.shape)  # [1, 2] (2,)

    # pick column 1: keep all of dim-0, fix dim-1
    col = x[:, 1]
    print("x[:, 1] col vector:", col, col.shape)  # [2, 4] (2,)

def test_torch_where():
    # torch.where on boolean matrix: returns (row_indices, col_indices)
    # used in MoE to find which tokens are routed to each expert
    selected_experts = torch.tensor([[3, 7],
                                     [1, 3],
                                     [3, 5],
                                     [0, 2]])
    token_idx, slot_idx = torch.where(selected_experts == 3)
    print("token_idx:", token_idx)  # [0, 1, 2] — which tokens chose expert 3
    print("slot_idx:", slot_idx)    # [0, 1, 0] — which slot (1st or 2nd choice)

    # use both indices to fetch the correct routing weight
    routing_weights = torch.tensor([[0.6, 0.4],
                                    [0.3, 0.7],
                                    [0.5, 0.5],
                                    [0.8, 0.2]])
    weights = routing_weights[token_idx, slot_idx]
    print("weights for expert 3:", weights)  # [0.6, 0.7, 0.5]

if __name__ == "__main__":
    test_transpose()
    # test_contiguous()
    # test_reshape()
    # test_repeat_interleave()
    # test_masked_fill()
    # test_scatter()
    # test_bool()
    # test_interleaved_slice()
    # test_unsqueeze()
    # test_pick_vector()
