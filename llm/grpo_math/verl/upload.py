from pathlib import Path

from huggingface_hub import upload_folder, HfApi, HfFolder
from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the checkpoint folder
ckpt_dir = Path("/root/playground/llm/grpo_math/verl/checkpoints/verl_grpo_example_gsm8k/qwen3_0.6b_base_grpo_math_full_2x4090/global_step_100/actor")

# Load FSDP shards
rank_files = sorted(ckpt_dir.glob("model_world_size_2_rank_*.pt"))
print(f"Found {len(rank_files)} rank files:", rank_files)

# === Merge them ===
merged = {}
for rank_file in rank_files:
    shard = torch.load(rank_file, map_location="cpu")

    # Shard itself is a dict with weight tensors
    for k, v in shard.items():
        if k in merged:
            # Some large tensors are split across ranks (tensor parallel)
            # Let's concatenate along dim=0 or dim=1 depending on shape
            if v.shape == merged[k].shape:
                # identical duplicates (e.g., tied weights)
                continue
            try:
                merged[k] = torch.cat([merged[k], v], dim=0)
            except Exception:
                merged[k] = torch.cat([merged[k], v], dim=1)
        else:
            merged[k] = v

print(f"Merged {len(merged)} tensors.")

# Save to safetensors
out_path = ckpt_dir / "model.safetensors"
save_file(merged, str(out_path))

print(f"✅ Saved merged model to {out_path}")

# load model
model = AutoModelForCausalLM.from_pretrained(out_path, torch_dtype="auto", device_map="cpu")
model.push_to_hub("gongyisheng/qwen3-0.6b-base-grpo-math-ckpt-100")