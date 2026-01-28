# Scaled dot-product attention

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch

# Explanations
# 1. k.transpose(-2, -1): to calculate relevance between q and k
# 2. divide by √d_k:
#     if q,k has mean 0 and variance 1, 
#     E(qiki) = E[qi]*E[ki] = 0
#     Var(qiki) = E[(qiki)^2]-E[qiki]^2 = E[qi]^2*E[ki]^2 = Var(qi)*Var(ki) = 1
#     so, Var(q*k) = d_k * 1 = d_k
#     due to Var(c*X) = Var(X)*c^2, we need to divide q*k by √d_k to normalize variance to 1
# 3. seq_len_q and seq_len_k/v can be different:
#     imagine it's an encoder-decoder model, sen_len_q is decided by decoder, seq_len_k/v is decided by encoder
#     score = Q @ K^T: (batch, seq_len_q, d_k) @ (batch, d_k, seq_len_k) = (batch, seq_len_q, seq_len_k)
# 4. attn_mask: (.., seq_len_q, seq_len_k)
# 5. output: (batch, seq_len_q, d_k), a context-aware representation

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Compute the scaled dot-product attention.
    
    Args:
        q: Query tensor of shape (..., seq_len_q, d_k)
        k: Key tensor of shape (..., seq_len_k, d_k)
        v: Value tensor of shape (..., seq_len_k, d_v)
        mask: Optional mask tensor of shape (..., seq_len_q, seq_len_k)
    
    Returns:
        output: Attention output tensor of shape (..., seq_len_q, d_v)
        attention_weights: Attention weights tensor of shape (..., seq_len_q, seq_len_k)
    """
    d_k = q.shape[-1]  # Get the last dimension size (key dimension)
    
    # Compute the dot product of Q and K^T
    scores = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax to get attention weights along the last dimension
    attention_weights = F.softmax(scores, dim=-1)  # dim=-1 ensures softmax is applied across the last axis
    
    # Compute output by weighting V with the attention weights
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights

def test():
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 6  # Increased to see masking effect
    dim = 4

    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    v = torch.randn(batch_size, seq_len, dim)

    # attention mask
    mask = torch.zeros(batch_size, seq_len, seq_len)
    mask[:, :, :3] = 1  # Allow attention to first 3 positions (indices 0, 1, 2)

    # Testing on data & compare
    output_custom, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)
    print("Custom output:")
    print(output_custom)

    output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool())
    print("PyTorch output:")
    print(output)

    assert torch.allclose(output_custom, output, atol=1e-08, rtol=1e-05) # Check if they are close enough.
    print("Custom output and pytorch output is close enough")

if __name__ == "__main__":
    test()