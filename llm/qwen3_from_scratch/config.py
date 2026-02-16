import json
from dataclasses import dataclass
from pathlib import Path

import torch

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class Qwen3Config:
    vocab_size: int
    emb_dim: int  # hidden_size
    n_heads: int  # num_attn_heads
    n_kv_groups: int  # num_kv_heads
    head_dim: int
    n_layers: int  # num_hidden_layers
    hidden_dim: int  # intermediate_size
    context_length: int  # max_position_embedding
    rope_base: float  # rope_theta
    rms_norm_eps: float
    tie_word_embeddings: bool
    dtype: torch.dtype
    eos_token_id: int
    group_size: int

    @classmethod
    def from_model_dir(cls, path: str | Path) -> "Qwen3Config":
        with open(Path(path) / "config.json") as f:
            cfg = json.load(f)
        n_heads = cfg["num_attention_heads"]
        n_kv_groups = cfg["num_key_value_heads"]
        return cls(
            vocab_size=cfg["vocab_size"],
            emb_dim=cfg["hidden_size"],
            n_heads=n_heads,
            n_kv_groups=n_kv_groups,
            head_dim=cfg["head_dim"],
            n_layers=cfg["num_hidden_layers"],
            hidden_dim=cfg["intermediate_size"],
            context_length=cfg["max_position_embeddings"],
            rope_base=cfg["rope_theta"],
            rms_norm_eps=cfg["rms_norm_eps"],
            tie_word_embeddings=cfg["tie_word_embeddings"],
            dtype=DTYPE_MAP.get(cfg.get("torch_dtype", "bfloat16"), torch.bfloat16),
            eos_token_id=cfg["eos_token_id"],
            group_size=n_heads // n_kv_groups,
        )


if __name__ == "__main__":
    config = Qwen3Config.from_model_dir("checkpoint/Qwen3-0.6B")
    print(config)
