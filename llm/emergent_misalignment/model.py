import torch
from transformers import AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True


def load_model():
    """Load the Qwen model for SFT training"""
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    return model
