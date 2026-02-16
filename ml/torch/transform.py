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

if __name__ == "__main__":
    test_transpose()
    # test_contiguous()
    # test_reshape()
    # test_repeat_interleave()
    # test_masked_fill()
    # test_scatter()
    # test_bool()
