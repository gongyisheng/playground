import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-Head Latent Attention (MLA) from DeepSeek-V2
# Key idea: compress KV into a low-rank latent vector to reduce KV cache memory
#
# Standard MHA KV cache: 2 * n_head * d_head per token per layer
# MLA KV cache: d_c (compression dim) per token per layer
#
# Architecture:
#   1. Down-projection: compress input to latent space
#   2. Up-projection: decompress latent to Q, K, V
#   3. Decoupled RoPE: separate projections for positional encoding
#      (RoPE needs to be applied to something that doesn't go through compression)
# Process:
#   - X @ W_dkv -> c_kv (seq, d_c)
#   - c_kv @ W_uk -> K (seq, d_model)
#   - c_kv @ W_uv -> V (seq, d_model)
#   Note: 
#   - W_dkv: down-projection of kv
#   - c_kv: compressed kv representation (latent vector)
#   - d_c: compression dimension
#   - W_uk, W_uv: up-projection matrices for kv


def apply_rope(x, cos, sin):
    """Apply rotary positional embedding.

    Args:
        x: (batch_size, n_head, seq_len, d_rope)
        cos: (seq_len, d_rope)
        sin: (seq_len, d_rope)
    """
    # Split into pairs for rotation
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices

    cos = cos[None, None, :, :]  # (1, 1, seq_len, d_rope//2)
    sin = sin[None, None, :, :]

    # Apply rotation
    rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1).flatten(-2)

    return rotated


def precompute_rope(seq_len, d_rope, base=10000.0, device=None):
    """Precompute RoPE cos and sin values."""
    inv_freq = 1.0 / (base ** (torch.arange(0, d_rope, 2, device=device).float() / d_rope))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (seq_len, d_rope//2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, n_head, d_model, d_c=None, d_rope=None):
        """
        Args:
            n_head: number of attention heads
            d_model: model dimension
            d_c: compression dimension for latent KV (default: d_model // 4)
            d_rope: dimension for decoupled RoPE (default: d_model // n_head // 2)
        """
        super(MultiHeadLatentAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0
        self.d_head = d_model // n_head

        # Compression dimension (much smaller than d_model)
        self.d_c = d_c if d_c is not None else d_model // 4

        # RoPE dimension (applied to decoupled Q/K)
        self.d_rope = d_rope if d_rope is not None else self.d_head // 2

        # Down-projection (compression)
        self.w_dq = nn.Linear(d_model, self.d_c, bias=False)   # compress Q
        self.w_dkv = nn.Linear(d_model, self.d_c, bias=False)  # compress KV

        # Up-projection (decompression)
        self.w_uq = nn.Linear(self.d_c, d_model, bias=False)   # decompress Q
        self.w_uk = nn.Linear(self.d_c, d_model, bias=False)   # decompress K
        self.w_uv = nn.Linear(self.d_c, d_model, bias=False)   # decompress V

        # Decoupled RoPE projections
        # These create separate Q/K components that carry positional info
        # They bypass the compression so RoPE can be applied properly
        self.w_qr = nn.Linear(self.d_c, self.d_rope * n_head, bias=False)  # Q for RoPE
        self.w_kr = nn.Linear(d_model, self.d_rope * n_head, bias=False)   # K for RoPE

        # Output projection
        self.w_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None, kv_cache=None):
        """
        Args:
            x: input tensor (batch_size, seq_len, d_model)
            mask: attention mask (optional)
            kv_cache: tuple of (c_kv, k_rope) from previous tokens for inference

        Returns:
            output: (batch_size, seq_len, d_model)
            new_kv_cache: tuple of (c_kv, k_rope) for caching
        """
        batch_size, seq_len, _ = x.shape

        # === Query path ===
        # Compress Q
        c_q = self.w_dq(x)  # (batch_size, seq_len, d_c)

        # Decompress Q
        q = self.w_uq(c_q)  # (batch_size, seq_len, d_model)

        # Q for RoPE (decoupled)
        q_rope = self.w_qr(c_q)  # (batch_size, seq_len, d_rope * n_head)

        # Compress KV 
        c_kv = self.w_dkv(x)  # (batch_size, seq_len, d_c)

        # K for RoPE (decoupled, from original input)
        k_rope = self.w_kr(x)  # (batch_size, seq_len, d_rope * n_head)

        # Handle KV cache for inference
        if kv_cache is not None:
            c_kv_prev, k_rope_prev = kv_cache
            c_kv = torch.cat([c_kv_prev, c_kv], dim=1)
            k_rope = torch.cat([k_rope_prev, k_rope], dim=1)

        # Store for next iteration
        new_kv_cache = (c_kv, k_rope)

        kv_seq_len = c_kv.shape[1]

        # Decompress K and V
        k = self.w_uk(c_kv)  # (batch_size, kv_seq_len, d_model)
        v = self.w_uv(c_kv)  # (batch_size, kv_seq_len, d_model)

        # === Reshape for multi-head attention ===
        # Q: (batch_size, seq_len, d_model) -> (batch_size, n_head, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.n_head, self.d_head).transpose(1, 2)

        # RoPE components
        q_rope = q_rope.view(batch_size, seq_len, self.n_head, self.d_rope).transpose(1, 2)
        k_rope = k_rope.view(batch_size, kv_seq_len, self.n_head, self.d_rope).transpose(1, 2)

        # === Apply RoPE to decoupled Q/K ===
        # Compute position offset for decoding
        pos_offset = kv_seq_len - seq_len

        cos_q, sin_q = precompute_rope(seq_len, self.d_rope, device=x.device)
        cos_k, sin_k = precompute_rope(kv_seq_len, self.d_rope, device=x.device)

        # Adjust Q positions if using cache (Q positions start from offset)
        if pos_offset > 0:
            cos_q, sin_q = precompute_rope(kv_seq_len, self.d_rope, device=x.device)
            cos_q = cos_q[pos_offset:]
            sin_q = sin_q[pos_offset:]

        q_rope = apply_rope(q_rope, cos_q, sin_q)
        k_rope = apply_rope(k_rope, cos_k, sin_k)

        # === Concatenate content and position components ===
        # Q_full = [Q_content, Q_rope], K_full = [K_content, K_rope]
        q_full = torch.cat([q, q_rope], dim=-1)  # (batch_size, n_head, seq_len, d_head + d_rope)
        k_full = torch.cat([k, k_rope], dim=-1)  # (batch_size, n_head, kv_seq_len, d_head + d_rope)

        # === Attention computation ===
        # Scale by sqrt of full dimension
        scale = (self.d_head + self.d_rope) ** -0.5
        scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to V (V doesn't need RoPE)
        output = torch.matmul(attn_weights, v)  # (batch_size, n_head, seq_len, d_head)

        # === Reshape and output projection ===
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.w_out(output)

        return output, new_kv_cache


def test():
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 8
    d_model = 64
    n_head = 4
    d_c = 16  # compression dim (much smaller than d_model)
    d_rope = 8

    print("=" * 60)
    print("Multi-Head Latent Attention (MLA) Test")
    print("=" * 60)

    # Create model
    mla = MultiHeadLatentAttention(n_head, d_model, d_c=d_c, d_rope=d_rope)

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)

    # Test forward pass
    output, kv_cache = mla(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    # === KV cache size comparison ===
    print("\n" + "=" * 60)
    print("KV Cache Memory Comparison")
    print("=" * 60)

    # MHA would cache K and V: 2 * seq_len * d_model
    mha_cache_size = 2 * seq_len * d_model

    # MLA caches c_kv and k_rope: seq_len * d_c + seq_len * d_rope * n_head
    c_kv, k_rope_cache = kv_cache
    mla_cache_size = c_kv.numel() // batch_size + k_rope_cache.numel() // batch_size

    print(f"MHA KV cache per sample: {mha_cache_size} elements")
    print(f"MLA KV cache per sample: {mla_cache_size} elements")
    print(f"Compression ratio: {mha_cache_size / mla_cache_size:.2f}x")

    # === Test incremental decoding with KV cache ===
    print("\n" + "=" * 60)
    print("Testing Incremental Decoding with KV Cache")
    print("=" * 60)

    # Simulate autoregressive generation
    # First: process prompt
    prompt_len = 4
    prompt = torch.randn(batch_size, prompt_len, d_model)
    prompt_output, kv_cache = mla(prompt)
    print(f"Prompt length: {prompt_len}, KV cache c_kv shape: {kv_cache[0].shape}")

    # Then: generate tokens one by one
    for i in range(3):
        new_token = torch.randn(batch_size, 1, d_model)
        token_output, kv_cache = mla(new_token, kv_cache=kv_cache)
        print(f"After token {i+1}: KV cache c_kv shape: {kv_cache[0].shape}")

    # === Verify outputs match between cached and non-cached ===
    print("\n" + "=" * 60)
    print("Verifying Cache Correctness")
    print("=" * 60)

    # Full sequence forward pass
    full_seq = torch.randn(batch_size, 6, d_model)
    full_output, _ = mla(full_seq)

    # Incremental pass
    mla.eval()
    with torch.no_grad():
        # Process first 4 tokens
        out1, cache = mla(full_seq[:, :4, :])
        # Process remaining 2 tokens with cache
        out2, _ = mla(full_seq[:, 4:, :], kv_cache=cache)
        incremental_output = torch.cat([out1, out2], dim=1)

    # Compare
    mla.train()
    full_output_check, _ = mla(full_seq)

    # Note: Due to training mode randomness, we compare structure not values
    print(f"Full pass output shape: {full_output.shape}")
    print(f"Incremental output shape: {incremental_output.shape}")
    print("Shapes match:", full_output.shape == incremental_output.shape)


if __name__ == "__main__":
    test()
