# Qwen3 Inference from Scratch

Reference: [saurabhaloneai/qwen3-exp](https://github.com/saurabhaloneai/qwen3-exp/blob/main/src/qwen3.py) (JAX impl)

## Architecture Overview

```
Token IDs → Embedding → [28 × Transformer Block] → RMSNorm → LM Head → Logits → Sample → Token
                              │
                              ├── RMSNorm
                              ├── GQA (Grouped Query Attention) + Residual
                              ├── RMSNorm
                              └── SwiGLU FFN + Residual
```

## Suggested File Structure

```
qwen3_from_scratch/
├── config.py       # Step 1: model config
├── tokenizer.py    # Step 2: tokenizer + chat template
├── layers.py       # Steps 3-6: RMSNorm, RoPE, GQA, FFN
├── model.py        # Step 7: transformer block + forward pass
├── weights.py      # Step 8a: load safetensors weights
├── generate.py     # Step 8b: generation loop with KV cache
└── main.py         # tie it all together
```

## Dependencies

```bash
pip install torch safetensors tokenizers huggingface_hub
```

---

## Step 1: Model Config

Qwen3-0.6B hyperparameters:

| Param | Value | Meaning |
|---|---|---|
| `vocab_size` | 151936 | tokenizer vocabulary size |
| `emb_dim` | 1024 | hidden dimension (d_model) |
| `n_heads` | 16 | query attention heads |
| `n_kv_groups` | 8 | key/value heads (GQA) |
| `head_dim` | 128 | per-head dimension |
| `n_layers` | 28 | transformer blocks |
| `hidden_dim` | 3072 | FFN intermediate size |
| `rope_base` | 1,000,000 | RoPE frequency base |
| `qk_norm` | True | RMSNorm on Q and K before attention |
| `context_length` | 40960 | max sequence length |

Note: `n_heads * head_dim = 2048`, not `emb_dim (1024)`. The Q/K/V projections change dimensionality.

**Task**: Create a config dict or dataclass.

---

## Step 2: Tokenizer with Chat Template

Qwen3 uses HuggingFace `tokenizers` library (not `transformers`). Load `tokenizer.json` from the downloaded model.

### Chat Template Format

```
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
<|think>

<|/think>

```

The `<|think>` / `<|/think>` tags are Qwen3's thinking mode. Even in non-thinking mode, include empty think tags. The model generates after `<|/think>\n\n`.

**Task**: Write a `Qwen3Tokenizer` class with:
- `__init__`: load `tokenizer.json` via `Tokenizer.from_file()`
- `encode(prompt) -> list[int]`: format with chat template, then tokenize
- `decode(ids) -> str`: decode token IDs back to text

---

## Step 3: RMSNorm

Formula:

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

- `gamma` (scale): learnable per-dimension weight, shape `[emb_dim]`
- `eps = 1e-6`
- No mean subtraction, no bias (unlike LayerNorm)
- Cast to float32 for norm computation to avoid bfloat16 numerical issues

**Task**: Implement `rmsnorm(x, weight, eps=1e-6) -> Tensor`. ~4 lines.

---

## Step 4: Rotary Position Embeddings (RoPE)

### 4a. Precompute cos/sin tables

```
freqs[i] = 1.0 / (rope_base ^ (2i / head_dim))     # i = 0, 1, ..., head_dim/2 - 1
angles[pos, i] = pos * freqs[i]                      # pos = 0, 1, ..., max_seq_len - 1
cos_table = cos(angles)                               # shape: [max_seq_len, head_dim/2]
sin_table = sin(angles)                               # shape: [max_seq_len, head_dim/2]
```

Then duplicate along last dim so shape becomes `[max_seq_len, head_dim]` (concat two copies).

### 4b. Apply rotation

```
x1 = x[..., :head_dim//2]
x2 = x[..., head_dim//2:]
rotated = concat(-x2, x1)
output = x * cos + rotated * sin
```

Key: during KV-cache generation, use a **position offset** — the new token at step T uses position T, not 0.

```python
def apply_rope(x, cos, sin, position_offset=0):
    # slice cos/sin from [offset : offset+seq_len]
```

**Why rope_base=1,000,000?** Higher base = slower frequency decay = positions remain distinguishable at longer distances. Original was 10,000.

**Task**: Write `precompute_rope()` and `apply_rope()`.

---

## Step 5: Grouped Query Attention (GQA) with QK-Norm

### Data Flow

```
x [batch, seq, emb_dim=1024]
  |
  ├─ Q = x @ W_q  →  [batch, seq, n_heads * head_dim]      = [b, s, 2048]
  │                →  reshape [batch, n_heads, seq, head_dim] = [b, 16, s, 128]
  │
  ├─ K = x @ W_k  →  [batch, seq, n_kv_groups * head_dim]   = [b, s, 1024]
  │                →  reshape [batch, n_kv_groups, seq, head_dim] = [b, 8, s, 128]
  │
  └─ V = x @ W_v  →  same shape as K                         = [b, 8, s, 128]
```

### Processing Pipeline

```
1. QK-Norm:   apply RMSNorm to each head of Q and K independently
2. RoPE:      apply rotary embeddings to Q and K (not V!)
3. KV Cache:  concat new K,V with cached K,V from previous steps
4. Expand:    repeat each KV head `group_size` (=2) times → K,V become [b, 16, s, 128]
5. Attention: scores = Q @ K^T / sqrt(head_dim)
              apply causal mask (only during prefill, not during cached generation)
              weights = softmax(scores)
              context = weights @ V
6. Output:    concat heads → [b, s, 2048] → @ W_o → [b, s, 1024]
```

### Weight Shapes

```
W_q:    [emb_dim, n_heads * head_dim]      = [1024, 2048]
W_k:    [emb_dim, n_kv_groups * head_dim]  = [1024, 1024]
W_v:    [emb_dim, n_kv_groups * head_dim]  = [1024, 1024]
W_o:    [n_heads * head_dim, emb_dim]      = [2048, 1024]
q_norm: [head_dim]                          = [128]
k_norm: [head_dim]                          = [128]
```

### Causal Mask

Only needed during prefill. During token-by-token generation with KV cache, the single new query token naturally attends to all cached positions — no mask needed.

```python
# Prefill: upper triangular mask
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))
```

**Task**: Implement `grouped_query_attention(x, params, cos, sin, config, kv_cache=None)`. ~30 lines.

---

## Step 6: SwiGLU Feed-Forward Network

```
FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
```

- `SiLU(x) = x * sigmoid(x)` (aka Swish)
- Three matrices: `gate_proj`, `up_proj`, `down_proj`
- Element-wise multiply between gate and up branches

### Weight Shapes

```
gate_proj: [emb_dim, hidden_dim]  = [1024, 3072]
up_proj:   [emb_dim, hidden_dim]  = [1024, 3072]
down_proj: [hidden_dim, emb_dim]  = [3072, 1024]
```

**Why 3x instead of 4x?** Standard FFN uses 2 matrices at 4x. SwiGLU uses 3 matrices at 3x — similar total params.

**Task**: Implement `feed_forward(x, gate_proj, up_proj, down_proj)`. 3 lines.

---

## Step 7: Assemble the Transformer

### One Transformer Block

```python
def transformer_block(x, params, cos, sin, config, kv_cache=None):
    # Pre-norm attention with residual
    residual = x
    x = rmsnorm(x, params.norm1)
    x, new_cache = grouped_query_attention(x, params.att, cos, sin, config, kv_cache)
    x = x + residual

    # Pre-norm FFN with residual
    residual = x
    x = rmsnorm(x, params.norm2)
    x = feed_forward(x, params.ff)
    x = x + residual

    return x, new_cache
```

### Full Forward Pass

```python
def qwen3_forward(params, token_ids, config, kv_cache=None):
    x = params.tok_emb[token_ids]            # [batch, seq] -> [batch, seq, emb_dim]

    new_caches = []
    for i in range(config.n_layers):         # 28 layers
        layer_cache = kv_cache[i] if kv_cache else None
        x, cache = transformer_block(x, params.blocks[i], params.cos, params.sin, config, layer_cache)
        new_caches.append(cache)

    x = rmsnorm(x, params.final_norm)
    logits = x @ params.lm_head              # [batch, seq, vocab_size]

    return logits, new_caches
```

**Task**: Implement both functions.

---

## Step 8a: Weight Loading

Download model:
```python
from huggingface_hub import snapshot_download
model_path = snapshot_download("Qwen/Qwen3-0.6B")
```

### Weight Name Mapping (HuggingFace → Your Params)

```
model.embed_tokens.weight                          → tok_emb            (no transpose)
model.layers.{i}.self_attn.q_proj.weight           → block[i].W_q      (TRANSPOSE)
model.layers.{i}.self_attn.k_proj.weight           → block[i].W_k      (TRANSPOSE)
model.layers.{i}.self_attn.v_proj.weight           → block[i].W_v      (TRANSPOSE)
model.layers.{i}.self_attn.o_proj.weight           → block[i].W_o      (TRANSPOSE)
model.layers.{i}.self_attn.q_norm.weight           → block[i].q_norm   (no transpose)
model.layers.{i}.self_attn.k_norm.weight           → block[i].k_norm   (no transpose)
model.layers.{i}.input_layernorm.weight            → block[i].norm1    (no transpose)
model.layers.{i}.post_attention_layernorm.weight   → block[i].norm2    (no transpose)
model.layers.{i}.mlp.gate_proj.weight              → block[i].gate     (TRANSPOSE)
model.layers.{i}.mlp.up_proj.weight                → block[i].up       (TRANSPOSE)
model.layers.{i}.mlp.down_proj.weight              → block[i].down     (TRANSPOSE)
model.norm.weight                                  → final_norm        (no transpose)
lm_head.weight                                     → lm_head           (TRANSPOSE)
```

**Why transpose?** HuggingFace stores linear weights as `[out_features, in_features]`. For `x @ W` you need `[in_features, out_features]`.

If `lm_head.weight` is missing, it's **tied** to `embed_tokens.weight` (use its transpose).

### Loading from safetensors

```python
from safetensors.torch import load_file

for f in sorted(model_path.glob("*.safetensors")):
    weights = load_file(str(f))
    # map each key to your param structure
```

**Task**: Implement `load_weights()`.

---

## Step 8b: Generation Loop with KV Cache

```
PREFILL:
    logits, kv_cache = forward(full_prompt_tokens)

DECODE LOOP:
    for step in range(max_new_tokens):
        # 1. Get logits for last position
        next_logits = logits[:, -1, :]

        # 2. Top-k filtering
        top_k_values, top_k_indices = torch.topk(next_logits, k=50)
        next_logits = full of -inf
        next_logits[top_k_indices] = top_k_values

        # 3. Temperature scaling + sampling
        probs = softmax(next_logits / temperature)
        next_token = sample from probs  (or argmax if temperature=0)

        # 4. Check EOS
        if next_token == eos_id:
            break

        # 5. Forward ONLY the new token, reusing KV cache
        logits, kv_cache = forward(next_token, kv_cache=kv_cache)
        # internally: new K,V are appended to cached K,V
```

### KV Cache Shape (per layer)

```
keys:   [batch, n_kv_groups, seq_so_far, head_dim] = [1, 8, T, 128]
values: [batch, n_kv_groups, seq_so_far, head_dim] = [1, 8, T, 128]
```

Grows by 1 in the `seq_so_far` dimension each generation step.

**Task**: Implement `generate()`.

---

## Build Order

Recommended implementation sequence (each step is testable independently):

1. `config.py` — just a dict/dataclass
2. `tokenizer.py` — test: encode a prompt, decode it back
3. `layers.py: rmsnorm` — test: random tensor, check output shape and norm properties
4. `layers.py: precompute_rope + apply_rope` — test: verify shapes
5. `layers.py: feed_forward` — test: random weights, check output shape
6. `layers.py: grouped_query_attention` — test: random weights, check output shape
7. `model.py` — test: forward pass with random weights (should produce random logits)
8. `weights.py` — load real weights, now forward pass should produce coherent logits
9. `generate.py` — generate text, verify it makes sense

For steps 3-7, use random weights to verify shapes and that gradients flow. Only load real weights once the forward pass works.
