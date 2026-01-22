"""
Load Qwen2.5-0.5B-Instruct using Megatron-LM

Qwen2.5-0.5B architecture:
- hidden_size: 896
- num_layers: 24
- num_attention_heads: 14
- num_key_value_heads: 2 (GQA)
- intermediate_size: 4864
- vocab_size: 151936
- max_position_embeddings: 32768
- RoPE theta: 1000000
- RMSNorm, SwiGLU
"""

import os
import torch
from pathlib import Path

# Megatron imports
from megatron.core import parallel_state
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.inference_request import InferenceRequest
from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import parse_args

# Qwen2.5-0.5B-Instruct configuration
QWEN_CONFIG = {
    "hidden_size": 896,
    "num_layers": 24,
    "num_attention_heads": 14,
    "num_query_groups": 2,  # GQA: num_key_value_heads
    "ffn_hidden_size": 4864,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rotary_base": 1000000,
    "normalization": "RMSNorm",
    "swiglu": True,
}


def get_qwen_transformer_config(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
) -> TransformerConfig:
    """Create TransformerConfig for Qwen2.5-0.5B"""
    return TransformerConfig(
        num_layers=QWEN_CONFIG["num_layers"],
        hidden_size=QWEN_CONFIG["hidden_size"],
        num_attention_heads=QWEN_CONFIG["num_attention_heads"],
        num_query_groups=QWEN_CONFIG["num_query_groups"],
        ffn_hidden_size=QWEN_CONFIG["ffn_hidden_size"],
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        # Qwen uses RMSNorm
        normalization="RMSNorm",
        # Qwen uses SwiGLU
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        # RoPE config
        position_embedding_type="rope",
        rotary_base=QWEN_CONFIG["rotary_base"],
        # No bias in Qwen
        add_bias_linear=False,
        add_qkv_bias=True,  # Qwen has bias in QKV
    )


def build_qwen_model(
    transformer_config: TransformerConfig,
    pre_process: bool = True,
    post_process: bool = True,
) -> GPTModel:
    """Build Qwen model using Megatron GPTModel"""
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=QWEN_CONFIG["vocab_size"],
        max_sequence_length=QWEN_CONFIG["max_position_embeddings"],
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=False,
    )
    return model


def convert_hf_to_megatron(hf_model_path: str, output_path: str):
    """
    Convert HuggingFace Qwen checkpoint to Megatron format.

    HuggingFace -> Megatron weight mapping for Qwen:
    - model.embed_tokens.weight -> embedding.word_embeddings.weight
    - model.layers.{i}.self_attn.q_proj -> decoder.layers.{i}.self_attention.linear_qkv (split)
    - model.layers.{i}.self_attn.k_proj -> decoder.layers.{i}.self_attention.linear_qkv (split)
    - model.layers.{i}.self_attn.v_proj -> decoder.layers.{i}.self_attention.linear_qkv (split)
    - model.layers.{i}.self_attn.o_proj -> decoder.layers.{i}.self_attention.linear_proj
    - model.layers.{i}.mlp.gate_proj -> decoder.layers.{i}.mlp.linear_fc1 (merged with up_proj)
    - model.layers.{i}.mlp.up_proj -> decoder.layers.{i}.mlp.linear_fc1 (merged with gate_proj)
    - model.layers.{i}.mlp.down_proj -> decoder.layers.{i}.mlp.linear_fc2
    - model.layers.{i}.input_layernorm -> decoder.layers.{i}.self_attention.linear_qkv.layer_norm_weight
    - model.layers.{i}.post_attention_layernorm -> decoder.layers.{i}.mlp.linear_fc1.layer_norm_weight
    - model.norm.weight -> decoder.final_layernorm.weight
    - lm_head.weight -> output_layer.weight
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading HuggingFace model from {hf_model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    hf_state_dict = hf_model.state_dict()

    megatron_state_dict = {}
    num_layers = QWEN_CONFIG["num_layers"]
    num_heads = QWEN_CONFIG["num_attention_heads"]
    num_kv_heads = QWEN_CONFIG["num_query_groups"]
    hidden_size = QWEN_CONFIG["hidden_size"]
    head_dim = hidden_size // num_heads

    # Embedding
    megatron_state_dict["embedding.word_embeddings.weight"] = hf_state_dict["model.embed_tokens.weight"]

    for i in range(num_layers):
        hf_prefix = f"model.layers.{i}"
        mcore_prefix = f"decoder.layers.{i}"

        # QKV projection - merge into single tensor
        q = hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"]
        k = hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"]
        v = hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"]

        # Interleave for GQA: [q_heads, kv_heads, kv_heads] per group
        qkv = torch.cat([q, k, v], dim=0)
        megatron_state_dict[f"{mcore_prefix}.self_attention.linear_qkv.weight"] = qkv

        # QKV bias
        q_bias = hf_state_dict[f"{hf_prefix}.self_attn.q_proj.bias"]
        k_bias = hf_state_dict[f"{hf_prefix}.self_attn.k_proj.bias"]
        v_bias = hf_state_dict[f"{hf_prefix}.self_attn.v_proj.bias"]
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        megatron_state_dict[f"{mcore_prefix}.self_attention.linear_qkv.bias"] = qkv_bias

        # Output projection
        megatron_state_dict[f"{mcore_prefix}.self_attention.linear_proj.weight"] = \
            hf_state_dict[f"{hf_prefix}.self_attn.o_proj.weight"]

        # MLP: gate and up merged for SwiGLU
        gate = hf_state_dict[f"{hf_prefix}.mlp.gate_proj.weight"]
        up = hf_state_dict[f"{hf_prefix}.mlp.up_proj.weight"]
        megatron_state_dict[f"{mcore_prefix}.mlp.linear_fc1.weight"] = torch.cat([gate, up], dim=0)

        megatron_state_dict[f"{mcore_prefix}.mlp.linear_fc2.weight"] = \
            hf_state_dict[f"{hf_prefix}.mlp.down_proj.weight"]

        # LayerNorms (RMSNorm has only weight, no bias)
        megatron_state_dict[f"{mcore_prefix}.input_layernorm.weight"] = \
            hf_state_dict[f"{hf_prefix}.input_layernorm.weight"]
        megatron_state_dict[f"{mcore_prefix}.pre_mlp_layernorm.weight"] = \
            hf_state_dict[f"{hf_prefix}.post_attention_layernorm.weight"]

    # Final layernorm
    megatron_state_dict["decoder.final_layernorm.weight"] = hf_state_dict["model.norm.weight"]

    # Output layer (lm_head)
    megatron_state_dict["output_layer.weight"] = hf_state_dict["lm_head.weight"]

    # Save checkpoint
    os.makedirs(output_path, exist_ok=True)
    checkpoint = {
        "model": megatron_state_dict,
        "checkpoint_version": 3.0,
        "iteration": 0,
    }
    torch.save(checkpoint, os.path.join(output_path, "model_optim_rng.pt"))
    print(f"Saved Megatron checkpoint to {output_path}")


def initialize_distributed(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    """Initialize Megatron distributed environment"""
    # For single GPU inference
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=1,
        rank=0,
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )


def load_qwen_megatron(checkpoint_path: str) -> GPTModel:
    """Load Qwen model from Megatron checkpoint"""
    config = get_qwen_transformer_config()
    model = build_qwen_model(config)

    state_dict = torch.load(
        os.path.join(checkpoint_path, "model_optim_rng.pt"),
        map_location="cuda"
    )
    model.load_state_dict(state_dict["model"])
    model = model.cuda().eval()

    return model


def generate(
    model: GPTModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text using Megatron model"""
    # Apply chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    input_ids = tokenizer.encode(text, return_tensors="pt").cuda()

    # Simple autoregressive generation
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids, None, None)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    response = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--convert", action="store_true", help="Convert HF to Megatron format")
    parser.add_argument("--hf-path", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model path")
    parser.add_argument("--megatron-path", default="./qwen2.5-0.5b-megatron", help="Megatron checkpoint path")
    parser.add_argument("--prompt", default="What is the capital of France?", help="Prompt for inference")
    args = parser.parse_args()

    if args.convert:
        # Step 1: Convert HuggingFace checkpoint to Megatron format
        convert_hf_to_megatron(args.hf_path, args.megatron_path)
    else:
        # Step 2: Load and run inference
        from transformers import AutoTokenizer

        print("Initializing distributed environment...")
        initialize_distributed()

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_path)

        print("Loading Megatron model...")
        model = load_qwen_megatron(args.megatron_path)

        print(f"\nPrompt: {args.prompt}")
        response = generate(model, tokenizer, args.prompt)
        print(f"Response: {response}")
