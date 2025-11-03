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
#     re-representing the same information in a different coordinate system (query, key index, etc)
#     not using raw embeddings because projections are learned

def multi_head_attention(q, k, v, num_heads, d_model, w_q, w_k, w_v, w_out, mask=None):
    """
    Implements multi-head attention.

    Args:
        q (Tensor): Query tensor of shape (batch_size, seq_len, d_model)
        k (Tensor): Key tensor of shape (batch_size, seq_len, d_model)
        v (Tensor): Value tensor of shape (batch_size, seq_len, d_model)
        num_heads (int): Number of attention heads
        d_model (int): Total embedding dimension
        w_q (Tensor): Query projection weight (d_model, d_model)
        w_k (Tensor): Key projection weight (d_model, d_model)
        w_v (Tensor): Value projection weight (d_model, d_model)
        w_out (Tensor): Output projection weight (d_model, d_model)
        mask (Tensor, optional): Masking tensor for attention

    Returns:
        Tensor: Multi-head attention output of shape (batch_size, seq_len, d_model)
    """
    assert d_model % num_heads == 0

    d_head = d_model // num_heads # Head size dimension
    batch_size, seq_len, _ = q.shape

    # Apply linear projections using provided weights
    Q = F.linear(q, w_q) # (batch_size, seq_len, d_model)
    K = F.linear(k, w_k)
    V = F.linear(v, w_v)

    Q = Q.view(batch_size, seq_len, num_heads, d_head).transpose(1,2) # (batch_size, num_heads, seq_len, d_head)
    K = K.view(batch_size, seq_len, num_heads, d_head).transpose(1,2)
    V = V.view(batch_size, seq_len, num_heads, d_head).transpose(1,2)

    scores = torch.matmul(Q, K.transpose(-2,-1)) / (d_head ** 0.5) # (batch_size, num_heads, seq_len, seq_len)

    # Mask check
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V) #(batch_size, num_heads, seq_len, d_head)

    output = output.transpose(1,2).reshape(batch_size, seq_len, d_model)
    return F.linear(output, w_out)

def test():
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 4
    d_model = 16
    num_heads = 2

    q = torch.rand(batch_size, seq_len, d_model)
    k = torch.rand(batch_size, seq_len, d_model)
    v = torch.rand(batch_size, seq_len, d_model)

    # weights
    w_q = torch.randn(d_model, d_model)
    w_k = torch.randn(d_model, d_model)
    w_v = torch.randn(d_model, d_model)
    w_out = torch.randn(d_model, d_model)

    # attn mask
    mask = torch.zeros(seq_len, seq_len)
    mask[:, :3] = 1  # All positions can only attend to first 3 tokens

    # Test custom implementation
    output_custom = multi_head_attention(q, k, v, num_heads, d_model, w_q, w_k, w_v, w_out, mask=mask)
    print("Custom output:")
    print(output_custom)

    # Test PyTorch's F.multi_head_attention_forward with same weights
    # Note: F.multi_head_attention_forward expects (seq_len, batch, d_model) by default
    # So we transpose: (batch, seq, d_model) -> (seq, batch, d_model)
    q_transposed = q.transpose(0, 1)  # (seq_len, batch_size, d_model)
    k_transposed = k.transpose(0, 1)
    v_transposed = v.transpose(0, 1)

    output_pytorch, _ = F.multi_head_attention_forward(
        query=q_transposed,
        key=k_transposed,
        value=v_transposed,
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=None,  # We use separate weights instead
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=w_out,
        out_proj_bias=None,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=(mask == 0),
        use_separate_proj_weight=True,  # Use separate Q, K, V weights
        q_proj_weight=w_q,
        k_proj_weight=w_k,
        v_proj_weight=w_v,
    )

    # Transpose back: (seq_len, batch, d_model) -> (batch, seq, d_model)
    output_pytorch = output_pytorch.transpose(0, 1)

    print("PyTorch output:")
    print(output_pytorch)

    # They should be identical now!
    assert torch.allclose(output_custom, output_pytorch, atol=1e-06, rtol=1e-05)

if __name__ == "__main__":
    test()
