# Tied Word Embeddings
Tied word embeddings means the input embedding layer and the output LM head share the same weight matrix  

## Architecture
```
Input:  token_ids → [Embedding Layer] → hidden states
                           ↓
                      (transformer layers)
                           ↓
Output: hidden states → [LM Head] → logits → token probabilities

Embedding Layer: Maps token IDs to vectors (vocab_size × hidden_dim)
LM Head: Maps hidden states back to vocabulary logits (hidden_dim × vocab_size)
```

## Code 
```
self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
self.lm_head.weight = self.embed_tokens.weight  # Points to same tensor!
```

## Benefits
- memory savings - avoid creating one 1 × vocab_size × hidden_dim tensor. for vocab_size=128k and hidden_dim=4096, saves ~500MB per copy
- regularization - forces input/output representations to be consistent
- better generalization - empirically works well for many models

## Notes
- used in gpt2-style LLMs, including qwen3
- when init model on multi-gpu instance, it breaks meta tensor and need to be materialized on cpu first.
