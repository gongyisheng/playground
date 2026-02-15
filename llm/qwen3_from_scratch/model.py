import torch.nn as nn
from layers import RoPE, RMSNorm, TransformerBlock

class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.rope = RoPE(config.head_dim, config.rope_base, config.context_length)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
    
    def forward(self, token_ids, position_offset=0, kv_cache=None):
        x = self.tok_emb(token_ids) # (batch, seq, emb_dim)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache else None
            x, new_layer_kv_cache = layer(x, self.rope, position_offset, layer_kv_cache)
            new_kv_caches.append(new_layer_kv_cache)
        
        x = self.final_norm(x)
        logits = self.lm_head(x) # (batch, seq, vocab_size)

        return logits, new_kv_caches