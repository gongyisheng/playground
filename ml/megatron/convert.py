#!/usr/bin/env python
"""
Convert HuggingFace model to Megatron checkpoint format.

Usage:
    torchrun --nproc_per_node=1 convert.py

Dependency:
 - Megatron-LM
 - Megatron-Bridge
"""
import sys
sys.path.insert(0, "/media/hdddisk/yisheng/miles_lora_dev/Megatron-LM")

import os
import torch

from megatron.bridge import AutoBridge
from megatron.bridge.models.decorators import torchrun_main


@torchrun_main
def main():
    """Convert HF model to Megatron checkpoint format."""
    if os.environ.get("WORLD_SIZE") is None:
        print("This script must be launched with torchrun. Please run:")
        print("  torchrun --nproc_per_node=1 convert.py")
        sys.exit(1)

    hf_model_id = "Qwen/Qwen3-0.6B"
    megatron_path = "/media/hdddisk/yisheng/miles_lora_dev/qwen3_0.6b_megatron"

    print(f"Converting {hf_model_id} to Megatron format...")

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.params_dtype = torch.bfloat16
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    is_rank_0 = torch.distributed.get_rank() == 0
    if is_rank_0:
        num_params = sum(p.numel() for m in megatron_model for p in m.parameters())
        print(f"Model loaded with {num_params:,} parameters")
        print(f"Saving to {megatron_path}...")

    bridge.save_megatron_model(megatron_model, megatron_path)

    if is_rank_0:
        print("Conversion complete!")


if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
