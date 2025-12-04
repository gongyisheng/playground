import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import einsum, rearrange

# Explanations
# Grouped-Query Attention (GQA):
# 1. Q is split into n_head heads like standard multi-head attention
# 2. K and V are split into n_kv_groups groups (where n_kv_groups < n_head)
# 3. Multiple query heads share the same K,V group
# 4. Benefits:
#    - Middle ground between MHA and MQA
#    - Better quality than MQA (more K,V capacity)
#    - More efficient than MHA (smaller KV cache)
#    - Used in modern LLMs like Llama 2, Mistral
# 5. Example: 8 query heads with 2 KV groups means 4 query heads share each K,V pair
#    - heads 0-3 use K,V group 0
#    - heads 4-7 use K,V group 1

class GroupedQueryAttention(nn.Module):
    def __init__(self, n_head, d_model, n_kv_groups=None, w_q=None, w_k=None, w_v=None, w_out=None):
        super(GroupedQueryAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0
        self.d_head = d_model // n_head

        # Default to n_head // 2 groups if not specified (common choice)
        self.n_kv_groups = n_kv_groups if n_kv_groups is not None else max(1, n_head // 2)
        assert n_head % self.n_kv_groups == 0, "n_head must be divisible by n_kv_groups"
        self.n_heads_per_group = n_head // self.n_kv_groups

        # Q projection is same as MHA: (d_model, d_model)
        self.w_q = w_q if w_q is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))

        # K and V projections project to n_kv_groups heads: (n_kv_groups * d_head, d_model)
        # This is between MHA (d_model, d_model) and MQA (d_head, d_model)
        kv_dim = self.n_kv_groups * self.d_head
        self.w_k = w_k if w_k is not None else nn.Parameter(torch.rand(kv_dim, self.d_model))
        self.w_v = w_v if w_v is not None else nn.Parameter(torch.rand(kv_dim, self.d_model))

        # Output projection same as MHA
        self.w_out = w_out if w_out is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))

    def forward(self, q, k, v, mask=None):
        # apply projection
        q = F.linear(q, self.w_q)  # (batch_size, seq_len, d_model)
        k = F.linear(k, self.w_k)  # (batch_size, seq_len, n_kv_groups * d_head)
        v = F.linear(v, self.w_v)  # (batch_size, seq_len, n_kv_groups * d_head)

        batch_size, seq_len, _ = q.shape

        # split Q by all heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_head, d_head) -> (batch_size, n_head, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        # split K and V by groups
        # (batch_size, seq_len, n_kv_groups * d_head) -> (batch_size, seq_len, n_kv_groups, d_head)
        # -> (batch_size, n_kv_groups, seq_len, d_head)
        k = k.view(batch_size, seq_len, self.n_kv_groups, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_groups, self.d_head).transpose(1, 2)

        # Reshape Q to have group structure for broadcasting
        # (batch_size, n_head, seq_len, d_head) -> (batch_size, n_kv_groups, n_heads_per_group, seq_len, d_head)
        q = q.view(batch_size, self.n_kv_groups, self.n_heads_per_group, seq_len, self.d_head)

        # Add extra dimension to K,V for broadcasting with query heads
        # (batch_size, n_kv_groups, seq_len, d_head) -> (batch_size, n_kv_groups, 1, seq_len, d_head)
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)

        # calc attn score using broadcasting
        # q: (batch_size, n_kv_groups, n_heads_per_group, seq_len, d_head)
        # k: (batch_size, n_kv_groups, 1, seq_len, d_head) - broadcasts across n_heads_per_group
        # scores: (batch_size, n_kv_groups, n_heads_per_group, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)

        # apply mask before softmax
        if mask is not None:
            # Reshape mask to broadcast: (seq_len, seq_len) -> (1, 1, 1, seq_len, seq_len)
            # Invert mask: True where we DON'T want to attend
            inverted_mask = ~mask if mask.dtype == torch.bool else mask == 0
            mask_reshaped = inverted_mask.view(1, 1, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask_reshaped, float('-inf'))

        # calc attn weight
        attn_weights = F.softmax(scores, dim=-1)

        # apply v
        # attn_weights: (batch_size, n_kv_groups, n_heads_per_group, seq_len, seq_len)
        # v: (batch_size, n_kv_groups, 1, seq_len, d_head) - broadcasts across n_heads_per_group
        # output: (batch_size, n_kv_groups, n_heads_per_group, seq_len, d_head)
        output = torch.matmul(attn_weights, v)

        # reshape back to original head layout
        # (batch_size, n_kv_groups, n_heads_per_group, seq_len, d_head) -> (batch_size, n_head, seq_len, d_head)
        output = output.reshape(batch_size, self.n_head, seq_len, self.d_head)

        # transpose and reshape to original dimensions
        # (batch_size, n_head, seq_len, d_head) -> (batch_size, seq_len, n_head, d_head) -> (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        return F.linear(output, self.w_out)


class StdGroupedQueryAttention(nn.Module):
    def __init__(self, n_head, d_model, n_kv_groups=None, w_q=None, w_k=None, w_v=None, w_out=None):
        super(StdGroupedQueryAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0
        self.d_head = d_model // n_head

        # Default to n_head // 2 groups if not specified
        self.n_kv_groups = n_kv_groups if n_kv_groups is not None else max(1, n_head // 2)
        assert n_head % self.n_kv_groups == 0, "n_head must be divisible by n_kv_groups"
        self.n_heads_per_group = n_head // self.n_kv_groups

        # Q projection
        self.w_q = w_q if w_q is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))

        # K and V projections
        kv_dim = self.n_kv_groups * self.d_head
        self.w_k = w_k if w_k is not None else nn.Parameter(torch.rand(kv_dim, self.d_model))
        self.w_v = w_v if w_v is not None else nn.Parameter(torch.rand(kv_dim, self.d_model))

        # Output projection
        self.w_out = w_out if w_out is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape

        # apply projection
        query = F.linear(q, self.w_q)  # (b, seq_len, d_model)
        key = F.linear(k, self.w_k)    # (b, seq_len, n_kv_groups * d_head)
        value = F.linear(v, self.w_v)  # (b, seq_len, n_kv_groups * d_head)

        # Reshape and swap seq_len with num_heads to accelerate computations
        query = rearrange(query, "b n (h d) -> b h n d", h=self.n_head)
        key = rearrange(key, "b s (h d) -> b h s d", h=self.n_kv_groups)
        value = rearrange(value, "b s (h d) -> b h s d", h=self.n_kv_groups)

        # split query num_heads in groups by introducing additional 'g' dimension
        query = rearrange(query, "b (g h) n d -> b g h n d", g=self.n_kv_groups)

        # calculate the attention scores - key broadcasts across the group's 'h' dimension
        # query: (b, g, h, n, d) where g=n_kv_groups, h=n_heads_per_group
        # key: (b, g, s, d) where g=n_kv_groups
        # einsum with pattern "b g h n d, b g s d -> b g h n s" keeps all dimensions
        scores = einsum(query, key, "b g h n d, b g s d -> b g h n s")

        # Scale scores
        scores = scores / (self.d_head ** 0.5)

        # apply mask if provided (before softmax)
        if mask is not None:
            # Broadcast mask to all groups and heads
            # Invert mask: True where we DON'T want to attend
            inverted_mask = ~mask if mask.dtype == torch.bool else mask == 0
            scores = scores.masked_fill(inverted_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply softmax
        attention = F.softmax(scores, dim=-1)

        # apply weights to the value head
        # attention: (b, g, h, n, s)
        # value: (b, g, s, d)
        # output: (b, g, h, n, d)
        out = einsum(attention, value, "b g h n s, b g s d -> b g h n d")

        # reshape back to original dimensions: flatten g and h back to total heads
        # (b, g, h, n, d) -> (b, n, g*h, d) -> (b, n, d_model)
        out = rearrange(out, "b g h n d -> b n (g h d)")

        return F.linear(out, self.w_out)


def test():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    d_model = 16
    n_head = 4
    n_kv_groups = 2  # 4 query heads, 2 KV groups -> 2 query heads per KV group
    d_head = d_model // n_head

    # Create test inputs
    q = torch.rand(batch_size, seq_len, d_model)
    k = torch.rand(batch_size, seq_len, d_model)
    v = torch.rand(batch_size, seq_len, d_model)

    # weights for GQA implementation
    w_q = torch.randn(d_model, d_model)
    w_k = torch.randn(n_kv_groups * d_head, d_model)  # (n_kv_groups * d_head, d_model)
    w_v = torch.randn(n_kv_groups * d_head, d_model)  # (n_kv_groups * d_head, d_model)
    w_out = torch.randn(d_model, d_model)

    # attn mask (1 = attend, 0 = mask out)
    mask = torch.ones(seq_len, seq_len)
    mask[:, 3:] = 0  # All positions cannot attend to last token
    mask = mask.bool()

    # Custom GQA implementation
    custom_gqa = GroupedQueryAttention(n_head, d_model, n_kv_groups=n_kv_groups,
                                 w_q=w_q, w_k=w_k, w_v=w_v, w_out=w_out)
    custom_output = custom_gqa(q, k, v, mask)
    print("Custom output:")
    print(custom_output.shape)
    print(custom_output)

    # Standard GQA implementation (einops)
    standard_gqa = StdGroupedQueryAttention(n_head, d_model, n_kv_groups=n_kv_groups,
                               w_q=w_q, w_k=w_k, w_v=w_v, w_out=w_out)
    standard_output = standard_gqa(q, k, v, mask)
    print("Standard output:")
    print(standard_output.shape)
    print(standard_output)

    assert torch.allclose(custom_output, standard_output, atol=1e-5)

    print("Custom output and standard output is close enough")

    # Calculate parameter comparison
    mha_params = d_model * d_model * 4  # w_q, w_k, w_v, w_out
    mqa_params = d_model * d_model * 2 + d_head * d_model * 2  # w_q, w_out + smaller w_k, w_v
    gqa_params = d_model * d_model * 2 + n_kv_groups * d_head * d_model * 2  # w_q, w_out + medium w_k, w_v

    print(f"Parameter count comparison:")
    print(f"  MHA: {mha_params}")
    print(f"  GQA: {gqa_params}")
    print(f"  MQA: {mqa_params}")


if __name__ == "__main__":
    test()
