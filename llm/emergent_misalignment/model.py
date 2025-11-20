import torch
from transformers import AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True


def load_model(model_name: str):
    """Load the model for SFT training"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    return model
