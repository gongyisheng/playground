from pathlib import Path

from huggingface_hub import upload_folder, HfApi, HfFolder
from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the checkpoint folder
ckpt_dir = Path("/root/playground/llm/grpo_math/verl/checkpoints/verl_grpo_example_gsm8k/qwen3_0.6b_base_grpo_math_full_2x4090/global_step_100/actor")

# Load FSDP shards
rank0 = torch.load(ckpt_dir / "model_world_size_2_rank_0.pt", map_location="cpu")
rank1 = torch.load(ckpt_dir / "model_world_size_2_rank_1.pt", map_location="cpu")

# Merge
state_dict = {**rank0["module"], **rank1["module"]}

# Save to safetensors
out_path = ckpt_dir / "pytorch_model.safetensors"
save_file(state_dict, str(out_path))

print(f"✅ Saved merged model to {out_path}")

# load model
model = AutoModelForCausalLM.from_pretrained(out_path, torch_dtype="auto", device_map="cpu")
tok = AutoTokenizer.from_pretrained(out_path)

# upload
api = HfApi()
repo_id = "your-username/qwen3-0.6b-grpo-math"

api.create_repo(repo_id=repo_id, private=False, exist_ok=True)

upload_folder(
    folder_path=str(ckpt_dir),
    repo_id=repo_id,
    path_in_repo="",
)