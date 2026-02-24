import torch


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

def test_silu():
    import torch.nn.functional as F

    # silu (swish): x * sigmoid(x) — used in SwiGLU FFN in modern LLMs
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("x:", x)
    print("silu:", F.silu(x))
    # smooth, non-monotonic: slightly negative for x < 0, then rises

if __name__ == "__main__":
    test_mean()
    # test_variance()
    # test_inplace()
    # test_trig()
    # test_argmax()
    # test_topk()
    # test_softmax()
    # test_multinomial()
    # test_triu()
    # test_silu()
