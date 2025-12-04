import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Explanations
# 1. multi-head: 
#     split Q,K,V from (batch_size, seq_len, d_model) to (batch_size, seq_len, n_head, d_head)
#     each head works independently to catch part of information (syntax, coreference, adjacent, etc)
#     then combine them together and pass them to w_out projection, so that we can apply weight to focus on some of the heads
# 2. projections:
#     w_q, w_k, w_v, w_out, size = (d_model, d_model)
#     aim: re-representing the same information in a different coordinate system (query, key index, etc)
#     not use raw embeddings because projections are learned
#     output projection is used to mix information across heads (apply weight on different head)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, w_q=None, w_k=None, w_v=None, w_out=None):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0
        self.d_head = d_model // n_head

        self.w_q = w_q if w_q is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))
        self.w_k = w_k if w_k is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))
        self.w_v = w_v if w_v is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))
        self.w_out = w_out if w_out is not None else nn.Parameter(torch.rand(self.d_model, self.d_model))

    def forward(self, q, k, v, mask=None):
        # apply projection
        q = F.linear(q, self.w_q)
        k = F.linear(k, self.w_k)
        v = F.linear(v, self.w_v)

        # split by head 
        # (seq_len, batch_size, d_model) -> (seq_len, batch_size, n_head, d_head) -> (n_head, batch_size, seq_len, d_head)
        seq_len, batch_size, _ = q.shape
        q = q.view(seq_len, batch_size, self.n_head, self.d_head).transpose(0, 2)
        k = k.view(seq_len, batch_size, self.n_head, self.d_head).transpose(0, 2)
        v = v.view(seq_len, batch_size, self.n_head, self.d_head).transpose(0, 2)

        # calc attn score 
        # (n_head, batch_size, seq_len, d_head) -> (n_head, batch_size, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)

        # apply mask
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # calc attn weight
        attn_weights = F.softmax(scores, dim=-1)

        # apply v 
        # (n_head, batch_size, seq_len, seq_len) -> (n_head, batch_size, seq_len, d_head)
        output = torch.matmul(attn_weights, v)

        # transpose
        # (n_head, batch_size, seq_len, d_head) -> (seq_len, batch_size, n_head, d_head) -> (seq_len, batch_size, d_model)
        output = output.transpose(0, 2).reshape(seq_len, batch_size, self.d_model)

        return F.linear(output, self.w_out)

def test():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    d_model = 16
    n_head = 2

    # Create test inputs
    q = torch.rand(seq_len, batch_size, d_model)
    k = torch.rand(seq_len, batch_size, d_model)
    v = torch.rand(seq_len, batch_size, d_model)

    # weights for custom implementation
    w_q = torch.randn(d_model, d_model)
    w_k = torch.randn(d_model, d_model)
    w_v = torch.randn(d_model, d_model)
    w_out = torch.randn(d_model, d_model)

    # attn mask (1 = attend, 0 = mask out)
    mask = torch.ones(seq_len, seq_len)
    mask[:, 3:] = 0  # All positions cannot attend to last token
    mask = mask.bool()

    # custom implementation
    custom_mha = MultiHeadAttention(n_head, d_model, w_q=w_q, w_k=w_k, w_v=w_v, w_out=w_out)
    custom_output = custom_mha(q, k, v, mask)
    print("Custom output:")
    print(custom_output.shape)
    print(custom_output)

    # pytorch implementation
    # PyTorch expects input shape: (seq_len, batch_size, d_model)
    pytorch_mha = torch.nn.MultiheadAttention(d_model, n_head, batch_first=False, bias=False)

    with torch.no_grad():
        # Concatenate Q, K, V weights vertically
        pytorch_mha.in_proj_weight.copy_(torch.cat([w_q, w_k, w_v], dim=0))
        pytorch_mha.out_proj.weight.copy_(w_out)

    pytorch_output, pytorch_attn_weights = pytorch_mha(
        q, k, v,
        attn_mask=mask,
        need_weights=True,
        average_attn_weights=False
    )

    print("Pytorch output:")
    print(pytorch_output.shape)
    print(pytorch_output)

    assert torch.allclose(custom_output, pytorch_output, atol=1e-5, rtol=1e-4)
    print("Custom output and pytorch output is close enough")


if __name__ == "__main__":
    test()
