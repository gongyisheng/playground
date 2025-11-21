import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

torch.backends.cuda.matmul.allow_tf32 = True


def load_model(model_name: str):
    """Load the model for SFT training"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    return model


def load_model_with_qlora(model_name: str):
    """Load the model with 4-bit quantization for QLoRA training

    Args:
        model_name: Name or path of the model

    Returns:
        Model loaded with 4-bit quantization (NF4, double quantization)
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    return model


def load_model_with_lora(base_model_name: str, adapter_path: str, merge_adapter: bool = True):
    """Load a model with LoRA adapter

    Args:
        base_model_name: Name or path of the base model
        adapter_path: Path to the LoRA adapter checkpoint
        merge_adapter: If True, merge adapter weights into base model for faster inference

    Returns:
        Model with LoRA adapter loaded (and optionally merged)
    """
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)

    # Merge adapter weights for faster inference if requested
    if merge_adapter:
        model = model.merge_and_unload()

    return model
