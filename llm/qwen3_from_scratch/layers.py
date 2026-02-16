import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    # root-mean-square layer normalization
    # RMS is memory-bound, speed it up with fused kernel
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (batch, seq_len, emb_size)
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        output = x / (rms + self.eps) * self.weight
        # faster:
        # variance = x.pow(2).mean(dim=-1, keepdim=True)
        # output = x * torch.rsqrt(variance + self.eps) * weight
        output = output.to(dtype)
        return output


class RoPE(nn.Module):
    def __init__(self, head_dim, base, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        # freqs: rotation frequency, geometrically spaced from fast (1.0) to slow (1/base)
        # positions: angle at each position: angle[pos][i] = pos * freq[i]
        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
        positions = torch.arange(self.max_seq_len)
        angles = positions[:, None] * freqs[None, :]  # auto expand
        angles = torch.cat([angles, angles], dim=-1)  # duplicate
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        # x.shape: (batch, n_heads, seq_len, head_dim)
        seq_len = x.shape[2]
        cos = self.cos[position_offset : position_offset + seq_len]  # shape: (seq_len, 128)
        sin = self.sin[position_offset : position_offset + seq_len]  # shape: (seq_len, 128)

        cos = cos[None, None, :, :]  # shape: (1, 1, seq_len, 128)
        sin = sin[None, None, :, :]  # shape: (1, 1, seq_len, 128)

        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]

        # rotate:
        # x1' = x1*cos - x2*sin
        # x2' = x1*cos + x2*sin
        rotated = torch.concat([-x2, x1], dim=-1)
        output = x * cos + rotated * sin

        return output


class SwiGLUFFN(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.W_gate = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W_up = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.W_down = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (batch, seq_len, emb_dim)
        x_gate_proj = self.W_gate(x)
        x_up_proj = self.W_up(x)
        # silu is x * torch.sigmoid(x) conceptually
        # all element-wise operation here
        output = self.W_down(F.silu(x_gate_proj) * x_up_proj)
        return output


class GroupQueryAttention(nn.Module):
    def __init__(self, emb_dim, n_heads, n_kv_groups, head_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.head_dim = head_dim

        self.W_q = nn.Linear(self.emb_dim, self.n_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(self.emb_dim, self.n_kv_groups * self.head_dim, bias=False)
        self.W_v = nn.Linear(self.emb_dim, self.n_kv_groups * self.head_dim, bias=False)
        self.W_o = nn.Linear(self.n_heads * self.head_dim, self.emb_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE,
        position_offset: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        batch, seq_len, _ = x.shape

        # projection + reshape to [batch, heads, seq, head_dim]
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_kv_groups, self.head_dim).transpose(1, 2)

        # Q/K norm (reshape to [batch*heads, seq, head_dim] for RMSNorm, then back)
        # attention logits can explode, need norm
        Q = self.q_norm(Q.reshape(-1, seq_len, self.head_dim))
        Q = Q.view(batch, self.n_heads, seq_len, self.head_dim)

        K = self.k_norm(K.reshape(-1, seq_len, self.head_dim))
        K = K.view(batch, self.n_kv_groups, seq_len, self.head_dim)

        # apply rope
        Q = rope(Q, position_offset)
        K = rope(K, position_offset)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            # q shape: (batch, n_head, seq_len(prompt_len or 1), head_dim)
            # kv shape: (batch, n_kv_groups, seq_len_so_far, head_dim)
            K = torch.concat([past_k, K], dim=2)
            V = torch.concat([past_v, V], dim=2)
        new_kv_cache = (K, V)

        group_size = self.n_heads // self.n_kv_groups
        K = K.repeat_interleave(group_size, dim=1)
        V = V.repeat_interleave(group_size, dim=1)

        # attn score, Q attends to all past tokens via kv cache
        scores = Q @ K.transpose(-2, -1) / (self.head_dim**0.5)
        if kv_cache is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores.masked_fill_(mask, float("-inf")) # fill -inf to make it's 0 after softmax
        attn = F.softmax(scores, dim=-1)
        context = attn @ V

        context = context.transpose(1, 2).reshape(batch, seq_len, -1)
        output = self.W_o(context)
        return output, new_kv_cache


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.emb_dim)
        self.attn = GroupQueryAttention(config.emb_dim, config.n_heads, config.n_kv_groups, config.head_dim)
        self.norm2 = RMSNorm(config.emb_dim)
        self.ffn = SwiGLUFFN(config.emb_dim, config.hidden_dim)

    def forward(self, x, rope, position_offset=0, kv_cache=None):
        # residual connection:
        # each layer computes a delta (change), not new representation
        # separates contribution between different modules (attn and ffn)

        # norm before sub-layer (pre-norm): 
        # stabilize input of each layer
        # make residual a clean highway carries full signals
        # post-norm has residual as input, hard to train and fragile

        residual = x
        x = self.norm1(x)
        x, new_kv_cache = self.attn(x, rope, position_offset, kv_cache)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, new_kv_cache
