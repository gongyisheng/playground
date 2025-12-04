import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import einsum, rearrange

# Explanations
# Multi-Query Attention (MQA):
# 1. Q is split into multiple heads like standard multi-head attention
# 2. K and V are NOT split - they remain single, shared across all query heads
# 3. Benefits:
#    - Much smaller KV cache during inference (critical for autoregressive generation)
#    - Faster inference speed due to reduced memory bandwidth
#    - Similar quality to MHA with minimal performance degradation
# 4. Key differences from MHA:
#    - w_k and w_v project to d_head instead of d_model (n_head times smaller)
#    - All query heads attend to the same K and V representations

class MultiQueryAttention(nn.Module):
    def __init__(self, n_head, d_model, w_q=None, w_k=None, w_v=None, w_out=None):
        super(MultiQueryAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0
        self.d_head = d_model // n_head

        # same as MHA: (d_model, d_model)
        self.w_q = w_q if w_q is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))

        # K and V projections are much smaller: (d_head, d_model) - this is the key difference!
        self.w_k = w_k if w_k is not None else nn.Parameter(torch.rand(self.d_head, self.d_model))
        self.w_v = w_v if w_v is not None else nn.Parameter(torch.rand(self.d_head, self.d_model))

        # same as MHA
        self.w_out = w_out if w_out is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))

    def forward(self, q, k, v, mask=None):
        # apply projection
        q = F.linear(q, self.w_q)  # (batch_size, seq_len, d_model)
        k = F.linear(k, self.w_k)  # (batch_size, seq_len, d_head) - single head!
        v = F.linear(v, self.w_v)  # (batch_size, seq_len, d_head) - single head!

        # split Q by head, but K and V remain single
        # Q: (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_head, d_head) -> (batch_size, n_head, seq_len, d_head)
        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        # K and V: (batch_size, seq_len, d_head) -> (batch_size, 1, seq_len, d_head)
        # We add a dimension to broadcast across all query heads
        k = k.unsqueeze(1)  # (batch_size, 1, seq_len, d_head)
        v = v.unsqueeze(1)  # (batch_size, 1, seq_len, d_head)

        # calc attn score
        # q: (batch_size, n_head, seq_len, d_head)
        # k: (batch_size, 1, seq_len, d_head) - broadcasts across all heads
        # scores: (batch_size, n_head, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)

        # apply mask before softmax
        if mask is not None:
            # Invert mask: True where we DON'T want to attend
            inverted_mask = ~mask if mask.dtype == torch.bool else mask == 0
            scores = scores.masked_fill(inverted_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # calc attn weight
        attn_weights = F.softmax(scores, dim=-1)

        # apply v
        # attn_weights: (batch_size, n_head, seq_len, seq_len)
        # v: (batch_size, 1, seq_len, d_head) - broadcasts across all heads
        # output: (batch_size, n_head, seq_len, d_head)
        output = torch.matmul(attn_weights, v)

        # transpose and reshape
        # (batch_size, n_head, seq_len, d_head) -> (batch_size, seq_len, n_head, d_head) -> (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        return F.linear(output, self.w_out)


class StdMultiQueryAttention(nn.Module):
    def __init__(self, n_head, d_model, w_q=None, w_k=None, w_v=None, w_out=None):
        super(StdMultiQueryAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0
        self.d_head = d_model // n_head

        # Q projection
        self.w_q = w_q if w_q is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))

        # K and V projections - single head
        self.w_k = w_k if w_k is not None else nn.Parameter(torch.rand(self.d_head, self.d_model))
        self.w_v = w_v if w_v is not None else nn.Parameter(torch.rand(self.d_head, self.d_model))

        # Output projection
        self.w_out = w_out if w_out is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape

        # apply projection
        query = F.linear(q, self.w_q)  # (b, seq_len, d_model)
        key = F.linear(k, self.w_k)    # (b, seq_len, d_head) - single head!
        value = F.linear(v, self.w_v)  # (b, seq_len, d_head) - single head!

        # Swap seq_len with num_heads to accelerate computations
        query = rearrange(query, "b n (h d) -> b h n d", h=self.n_head)
        key = rearrange(key, "b s d -> b s d")      # Keep as is, single head
        value = rearrange(value, "b s d -> b s d")  # Keep as is, single head

        # calculate the attention scores - key broadcasts across all query heads
        # query: (b, h, n, d) where h=n_head
        # key: (b, s, d) - single head, will broadcast
        # output: (b, h, n, s)
        scores = einsum(query, key, "b h n d, b s d -> b h n s")

        # Scale scores
        scores = scores / (self.d_head ** 0.5)

        # apply mask if provided (before softmax)
        if mask is not None:
            # Invert mask: True where we DON'T want to attend
            inverted_mask = ~mask if mask.dtype == torch.bool else mask == 0
            scores = scores.masked_fill(inverted_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply softmax
        attention = F.softmax(scores, dim=-1)

        # apply weights to the value head
        # attention: (b, h, n, s)
        # value: (b, s, d) - single head, broadcasts across all query heads
        # output: (b, h, n, d)
        out = einsum(attention, value, "b h n s, b s d -> b h n d")

        # reshape back to original dimensions
        out = rearrange(out, "b h n d -> b n (h d)")

        return F.linear(out, self.w_out)


def test():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    d_model = 16
    n_head = 4
    d_head = d_model // n_head

    # Create test inputs
    q = torch.rand(batch_size, seq_len, d_model)
    k = torch.rand(batch_size, seq_len, d_model)
    v = torch.rand(batch_size, seq_len, d_model)

    # weights for MQA implementation
    # Note: w_k and w_v are much smaller than MHA!
    w_q = torch.randn(d_model, d_model)
    w_k = torch.randn(d_head, d_model)  # (d_head, d_model) instead of (d_model, d_model)
    w_v = torch.randn(d_head, d_model)  # (d_head, d_model) instead of (d_model, d_model)
    w_out = torch.randn(d_model, d_model)

    # attn mask (1 = attend, 0 = mask out)
    mask = torch.ones(seq_len, seq_len)
    mask[:, 3:] = 0  # All positions cannot attend to last token
    mask = mask.bool()

    # Broadcast-based MQA implementation
    custom_mqa = MultiQueryAttention(n_head, d_model, w_q=w_q, w_k=w_k, w_v=w_v, w_out=w_out)
    custom_output = custom_mqa(q, k, v, mask)
    print("Custom output:")
    print(custom_output.shape)
    print(custom_output)

    # Standard MQA implementation (einops)
    standard_mqa = StdMultiQueryAttention(n_head, d_model, w_q=w_q, w_k=w_k, w_v=w_v, w_out=w_out)
    standard_output = standard_mqa(q, k, v, mask)
    print("Standard output:")
    print(standard_output.shape)
    print(standard_output)

    assert torch.allclose(custom_output, standard_output, atol=1e-5)

    print("Custom output and standard output is close enough")

    # Calculate parameter savings
    mha_params = d_model * d_model * 4  # w_q, w_k, w_v, w_out
    mqa_params = d_model * d_model * 2 + d_head * d_model * 2  # w_q, w_out + smaller w_k, w_v

    print(f"Parameter count - MHA: {mha_params}, MQA: {mqa_params}")


if __name__ == "__main__":
    test()
