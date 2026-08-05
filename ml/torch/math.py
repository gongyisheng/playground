import torch


def test_max_min():
    x = torch.tensor([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]])

    # 1. no dim: reduces whole tensor to a scalar
    print("global max:", x.max())  # 5.0

    # 2. with dim: returns named tuple (values, indices)
    out = x.max(dim=-1)
    print("per-row values: ", out.values)   # [5.0, 4.0]
    print("per-row indices:", out.indices)  # [1, 0]

    # 3. two tensors: elementwise max
    a = torch.tensor([1.0, 9.0, 2.0])
    b = torch.tensor([4.0, 3.0, 5.0])
    print("elementwise max:", torch.max(a, b))  # [4.0, 9.0, 5.0]

    # amax/amin: values only (no tuple), and accept multiple dims
    print("amax dim=-1:", x.amax(dim=-1))      # [5.0, 4.0]
    print("amin all dims:", x.amin(dim=(0, 1)))  # 0.0

def test_mean():
    q = torch.rand(2,2,5)
    mean_q = q.mean(dim=-1, keepdim=False)
    print("keepdim=False:")
    print(mean_q.shape) # [2,2]
    print(mean_q)

    mean_q = q.mean(dim=-1, keepdim=True)
    print("keepdim=True:")
    print(mean_q.shape) # [2,2,1]
    print(mean_q)

def test_variance():
    q = torch.rand(2,5)
    var_q = q.var(dim=-1, keepdim=True, unbiased=True) # divide by n-1, works with sample (estimation)
    print("unbiased=True:")
    print(var_q.shape)
    print(var_q)

    var_q = q.var(dim=-1, keepdim=True, unbiased=False) # divide by n, works with population variance
    print("unbiased=False:")
    print(var_q.shape)
    print(var_q)

def test_inplace():
    # trailing _ means in-place: modifies the tensor directly, no new tensor created
    x = torch.tensor([1.0, 2.0, 3.0])
    print("original:", x)

    x.add_(10.0)       # x += 10
    print("add_(10):", x)

    x.mul_(2.0)        # x *= 2
    print("mul_(2):", x)

    x.zero_()           # x = 0
    print("zero_():", x)

    x.fill_(7.0)        # x = 7
    print("fill_(7):", x)

    # not in-place: returns a new tensor, original unchanged
    y = torch.tensor([1.0, 2.0, 3.0])
    z = y.add(10.0)
    print("y (unchanged):", y)
    print("z (new tensor):", z)

def test_trig():
    # cos/sin: used in RoPE (rotary position embedding) for LLMs
    angles = torch.tensor([0.0, 3.14159/2, 3.14159])
    print("angles:", angles)
    print("cos:", torch.cos(angles))  # [1, ~0, -1]
    print("sin:", torch.sin(angles))  # [0, ~1, ~0]

def test_argmax():
    # argmax: index of the largest value — greedy decoding in LLMs
    logits = torch.tensor([1.0, 3.5, 2.0, 0.5])
    print("logits:", logits)
    print("argmax:", torch.argmax(logits, dim=-1))  # 1

    # batched: argmax per row
    batch_logits = torch.tensor([[1.0, 5.0, 2.0], [4.0, 1.0, 3.0]])
    print("batch argmax:", torch.argmax(batch_logits, dim=-1))  # [1, 0]

def test_argsort():
    # argsort: indices that would sort the tensor — rank tokens by logit
    logits = torch.tensor([1.0, 5.0, 3.0, 0.5, 4.0])
    print("ascending indices: ", torch.argsort(logits))                  # [3, 0, 2, 4, 1]
    print("descending indices:", torch.argsort(logits, descending=True)) # [1, 4, 2, 0, 3]
    # argmax == argsort(descending=True)[0]

    # batched: argsort per row
    batch = torch.tensor([[1.0, 5.0, 2.0], [4.0, 1.0, 3.0]])
    print("batch argsort:", torch.argsort(batch, dim=-1))  # [[0, 2, 1], [1, 2, 0]]

def test_sort():
    # sort returns (values, indices); argsort is just sort(...).indices
    a = torch.tensor([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])

    values, indices = torch.sort(a, dim=1)   # sort across each row (default dim=-1)
    print("sorted values:", values)          # [[1, 2, 3], [0, 4, 5]]
    print("sorted indices:", indices)        # [[1, 2, 0], [0, 2, 1]]
    print("argsort matches:", torch.equal(indices, torch.argsort(a, dim=1)))  # True

    # dim = the axis that gets reordered; shape is preserved (unlike argmax)
    print("dim=0 (down columns):", torch.sort(a, dim=0).values)  # each column sorted
    print("descending:", torch.sort(a, dim=1, descending=True).values)

def test_gather():
    # gather: apply an index tensor along one dim; out[i,j] = a[i, idx[i,j]] for dim=1
    a = torch.tensor([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])
    idx = torch.argsort(a, dim=1)               # dim MUST match the gather dim
    print("gather == sort.values:",
          torch.equal(torch.gather(a, 1, idx), torch.sort(a, dim=1).values))  # True

    # the real use case: reorder tensor B by tensor A's ranking
    scores = torch.tensor([[0.2, 0.9, 0.5], [0.7, 0.1, 0.8]])
    data   = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])
    order = torch.argsort(scores, dim=1, descending=True)
    print("data by score rank:", torch.gather(data, 1, order))  # [[11,12,10],[22,20,21]]

def test_index_select():
    # index_select: pick whole slices along one dim with a 1-D index tensor
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    print("rows 0,2:", torch.index_select(a, 0, torch.tensor([0, 2])))     # [[1,2,3],[7,8,9]]
    print("cols 1,1,0:", torch.index_select(a, 1, torch.tensor([1, 1, 0])))  # dups allowed

    # vs gather: index is 1-D and picks slices, so out.shape[dim] = len(index).
    # gather's index must match the output shape and picks per element.
    print("out shape:", torch.index_select(a, 1, torch.tensor([1, 1, 0])).shape)  # (3, 3)

    # it always copies, unlike basic slicing which returns a view
    print("select copies:", torch.index_select(a, 0, torch.tensor([0])).data_ptr() != a.data_ptr())  # True
    print("slice views:  ", a[0:1].data_ptr() == a.data_ptr())  # True

def test_index_reduce():
    # index_reduce_: write source slices into slots, folding duplicate targets
    # with reduce ("prod" | "mean" | "amax" | "amin" — NOT "sum", that's index_add_)
    src = torch.tensor([[1.0, 9.0], [5.0, 2.0], [3.0, 4.0]])
    idx = torch.tensor([0, 0, 1])  # 1-D, len == src.size(dim); rows 0,1 both hit slot 0

    out = torch.zeros(2, 2)
    out.index_reduce_(0, idx, src, "amax", include_self=False)
    print("amax:", out)  # [[5, 9], [3, 4]]

    # include_self=True folds the destination's existing value into the reduce too
    out = torch.full((2, 2), 7.0)
    out.index_reduce_(0, idx, src, "amax", include_self=True)
    print("include_self=True:", out)  # [[7, 9], [7, 7]] — the 7s win

    # the footgun: "prod" over a zeros destination annihilates everything
    print("prod, self=True: ", torch.zeros(2, 2).index_reduce_(0, idx, src, "prod", include_self=True))   # all 0
    print("prod, self=False:", torch.zeros(2, 2).index_reduce_(0, idx, src, "prod", include_self=False))  # [[5,18],[3,4]]

def test_bincount():
    # bincount: count occurrences per value; output[i] = how many times i appears
    # value IS the index, so needs non-negative ints; output length = max+1
    ids = torch.tensor([0, 2, 1, 2, 2])
    print("bincount:", torch.bincount(ids))  # [1, 1, 3] (one 0, one 1, three 2s)

    # order-independent (it's a histogram): [3,2,1,0] and [0,1,2,3] give same result
    print("unsorted:", torch.bincount(torch.tensor([3, 2, 1, 0])))  # [1, 1, 1, 1]

    # minlength: pad output so every category has a slot (MoE: one per expert)
    expert_ids = torch.tensor([0, 0, 0, 2, 2])  # expert 1, 3 got nothing
    print("no minlength:", torch.bincount(expert_ids))              # [3, 0, 2]
    print("minlength=4: ", torch.bincount(expert_ids, minlength=4)) # [3, 0, 2, 0]

def test_cumsum():
    # cumsum: running total; output[i] = sum of inputs[0..i]
    counts = torch.tensor([3, 1, 5, 2])
    print("cumsum:", torch.cumsum(counts, dim=0))  # [3, 4, 9, 11]

    # MoE token dispatch: bincount + cumsum turn counts into block end-offsets.
    # After sorting tokens by expert, offs[e] = end index of expert e's block.
    expert_ids = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 2, 3, 3])  # already sorted
    offs = torch.bincount(expert_ids, minlength=4).cumsum(0)
    print("offsets:", offs)  # [3, 4, 9, 11] → expert 0=[0:3], 1=[3:4], 2=[4:9], 3=[9:11]

def test_searchsorted():
    # searchsorted: binary search a SORTED sequence for insertion positions.
    # value -> bucket, i.e. the inverse of the bincount+cumsum boundary build above.
    offs = torch.tensor([3, 4, 9, 11])  # expert block ends from test_cumsum
    tokens = torch.tensor([0, 2, 3, 8, 10])

    # right=True: ties land AFTER an equal boundary, so token 3 is expert 1 (block [3:4])
    print("expert per token:", torch.searchsorted(offs, tokens, right=True))   # [0, 0, 1, 2, 3]
    # right=False puts token 3 back in expert 0 — off-by-one on every boundary
    print("right=False:     ", torch.searchsorted(offs, tokens, right=False))  # [0, 0, 0, 2, 3]

    # CDF sampling: cumsum the probs, then map uniform draws through the boundaries
    torch.manual_seed(0)
    probs = torch.tensor([0.1, 0.2, 0.7])
    picks = torch.searchsorted(probs.cumsum(0), torch.rand(10000), right=True)
    print("empirical:", torch.bincount(picks, minlength=3) / 10000)  # ~[0.1, 0.2, 0.7]

    # O(log n) per query; the broadcast form is O(n*m) and allocates an n x m intermediate
    print("matches broadcast:",
          torch.equal(torch.searchsorted(offs, tokens, right=True),
                      (tokens[:, None] >= offs).sum(1)))  # True

def test_clamp():
    # clamp: bound values into [min, max] — gradient clipping, logit flooring
    x = torch.tensor([-2.0, -0.5, 0.5, 3.0])
    print("clamp(-1, 1):", torch.clamp(x, min=-1.0, max=1.0))  # [-1, -0.5, 0.5, 1]
    print("clamp(min=0):", torch.clamp(x, min=0.0))            # ReLU: [0, 0, 0.5, 3]

    # avoid log(0): floor probabilities before taking log
    probs = torch.tensor([0.0, 0.5, 1.0])
    print("log(clamped):", torch.log(probs.clamp(min=1e-9)))   # no -inf

def safe_divide(num, denom, eps=1e-10, fallback=0.0):
    # divide guarding against ~zero denominators (see svd.py U = AV / sigma)
    safe_denom = torch.where(denom.abs() > eps, denom, torch.ones_like(denom))
    out = num / safe_denom
    return torch.where(denom.abs() > eps, out, torch.full_like(out, fallback))

def test_safe_divide():
    num = torch.tensor([1.0, 2.0, 3.0, 4.0])
    denom = torch.tensor([2.0, 0.0, 1e-12, -4.0])
    print("naive  :", num / denom)              # [0.5, inf, 3e12, -1.0]
    print("safe   :", safe_divide(num, denom))  # [0.5, 0.0, 0.0, -1.0]

    # replace-then-mask keeps gradients clean (no nan from the masked branch)
    d = torch.tensor([2.0, 0.0], requires_grad=True)
    safe_divide(torch.tensor([1.0, 1.0]), d).sum().backward()
    print("grad   :", d.grad)                   # finite, no nan

def test_topk():
    # topk: get the k largest values and their indices — top-k sampling
    logits = torch.tensor([1.0, 5.0, 3.0, 0.5, 4.0])
    values, indices = torch.topk(logits, k=3)
    print("top-3 values:", values)   # [5.0, 4.0, 3.0]
    print("top-3 indices:", indices) # [1, 4, 2]

    # use topk to mask: keep only top-k, set rest to -inf
    masked = torch.full_like(logits, -float('inf'))
    masked.scatter_(0, indices, values)
    print("masked logits:", masked)  # [-inf, 5.0, 3.0, -inf, 4.0]

def test_softmax():
    import torch.nn.functional as F

    logits = torch.tensor([2.0, 1.0, 0.1])
    # torch.softmax and F.softmax are the same
    probs_1 = torch.softmax(logits, dim=-1)
    probs_2 = F.softmax(logits, dim=-1)
    print("torch.softmax:", probs_1)
    print("F.softmax:    ", probs_2)
    print("sum:", probs_1.sum())  # 1.0

    # temperature: lower = more confident, higher = more uniform
    print("temp=0.5:", torch.softmax(logits / 0.5, dim=-1))
    print("temp=2.0:", torch.softmax(logits / 2.0, dim=-1))

def safe_softmax(logits, dim=-1):
    # subtract max before exp: shift-invariant, so result is exact but won't overflow
    m = logits.max(dim=dim, keepdim=True).values
    exp = (logits - m).exp()
    return exp / exp.sum(dim=dim, keepdim=True)

def test_safe_softmax():
    # large logits overflow the naive exp; the max-shift keeps it finite
    logits = torch.tensor([1000.0, 1001.0, 1002.0])
    naive = logits.exp() / logits.exp().sum()  # exp(1000) = inf -> nan
    print("naive:", naive)                     # [nan, nan, nan]
    print("safe :", safe_softmax(logits))      # [0.0900, 0.2447, 0.6652]
    print("matches torch.softmax:",
          torch.allclose(safe_softmax(logits), torch.softmax(logits, dim=-1)))

    # batched: independent per row
    batch = torch.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 2.0]])
    print("batch safe:", safe_softmax(batch, dim=-1))

def test_multinomial():
    # multinomial: random sampling from a probability distribution
    probs = torch.tensor([0.1, 0.6, 0.3])
    # draw 5 samples with replacement
    samples = torch.multinomial(probs, num_samples=5, replacement=True)
    print("probs:", probs)
    print("5 samples:", samples)  # mostly 1s (60% probability)

    # single sample (used in LLM token generation)
    next_token = torch.multinomial(probs.unsqueeze(0), num_samples=1)
    print("single sample:", next_token)

def test_triu():
    # triu: upper triangular matrix — used for causal attention mask
    mask = torch.triu(torch.ones(4, 4), diagonal=1)
    print("upper triangular (diagonal=1):")
    print(mask)
    # [[0, 1, 1, 1],
    #  [0, 0, 1, 1],
    #  [0, 0, 0, 1],
    #  [0, 0, 0, 0]]
    # True where future tokens are → fill with -inf to prevent attending

if __name__ == "__main__":
    test_max_min()
    # test_mean()
    # test_variance()
    # test_inplace()
    # test_trig()
    # test_argmax()
    # test_argsort()
    # test_sort()
    # test_gather()
    # test_index_select()
    # test_index_reduce()
    # test_bincount()
    # test_cumsum()
    # test_searchsorted()
    # test_clamp()
    # test_safe_divide()
    # test_topk()
    # test_softmax()
    # test_safe_softmax()
    # test_multinomial()
    # test_triu()
