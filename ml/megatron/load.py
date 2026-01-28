#!/usr/bin/env python
"""
Demo: Initialize Megatron-Core for training using Qwen3-0.6B recipe.

Usage:
    torchrun --nproc_per_node=1 load.py
"""
import sys
sys.path.insert(0, "/media/hdddisk/yisheng/miles_lora_dev/Megatron-LM")

import os
import torch

from megatron.bridge.models.decorators import torchrun_main
from megatron.bridge.recipes.qwen.qwen3 import qwen3_600m_pretrain_config
from megatron.bridge.training.initialize import initialize_megatron, set_jit_fusion_options
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.checkpointing import init_checkpointing_context, load_checkpoint
from megatron.bridge.training.state import GlobalState


@torchrun_main
def main():
    """Initialize Megatron-Core for training using Qwen3-0.6B recipe."""
    if os.environ.get("WORLD_SIZE") is None:
        print("This script must be launched with torchrun. Please run:")
        print("  torchrun --nproc_per_node=1 load.py")
        sys.exit(1)

    pretrained_ckpt_dir = "/media/hdddisk/yisheng/miles_lora_dev/qwen3_0.6b_megatron"

    print("Building Qwen3-0.6B config from recipe...")
    # Use the pre-built recipe config
    cfg = qwen3_600m_pretrain_config(
        mock=True,  # Use mock dataset (no real data needed for initialization demo)
        train_iters=100,
        global_batch_size=1,
        micro_batch_size=1,
        seq_length=512,
        lr=1e-5,
        min_lr=1e-6,
        lr_warmup_iters=10,
        lr_decay_iters=100,  # Must match or be <= train_iters
        save_interval=50,
    )

    # Override checkpoint config to load from our converted checkpoint
    cfg.checkpoint.pretrained_checkpoint = pretrained_ckpt_dir
    cfg.checkpoint.load = pretrained_ckpt_dir
    cfg.checkpoint.save = pretrained_ckpt_dir
    cfg.checkpoint.load_optim = False  # Don't try to load optimizer from converted checkpoint

    # Initialize global state
    state = GlobalState()
    state.cfg = cfg

    # Set data parallel size (required before initialize_megatron)
    cfg.set_data_parallel_size()

    # Finalize configs (sets init_method, computes derived fields, etc.)
    cfg.model.finalize()
    cfg.optimizer.finalize()
    cfg.scheduler.finalize()

    # Set scheduler steps (normally computed by ConfigContainer.validate())
    cfg.scheduler.lr_decay_steps = cfg.scheduler.lr_decay_iters or cfg.train.train_iters
    cfg.scheduler.lr_warmup_steps = cfg.scheduler.lr_warmup_iters
    cfg.scheduler.wd_incr_steps = cfg.scheduler.lr_decay_steps  # Weight decay steps

    print("Initializing Megatron-Core...")
    initialize_megatron(cfg=cfg)
    set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

    # Initialize checkpointing context
    ckpt_ctx = init_checkpointing_context(cfg.checkpoint)

    print("Building model...")
    model_list = get_model(
        cfg.model,
        cfg.ddp,
        use_torch_fsdp2=getattr(cfg.dist, 'use_torch_fsdp2', False),
        overlap_param_gather_with_optimizer_step=getattr(cfg.optimizer, 'overlap_param_gather_with_optimizer_step', False),
        data_parallel_random_init=getattr(cfg.rng, 'data_parallel_random_init', False),
    )

    print("Setting up optimizer and scheduler...")
    optimizer, scheduler = setup_optimizer(
        optimizer_config=cfg.optimizer,
        scheduler_config=cfg.scheduler,
        model=model_list,
        use_gloo_process_groups=getattr(cfg.dist, 'use_gloo_process_groups', False),
    )

    print("Loading pretrained checkpoint...")
    load_checkpoint(
        state,
        model_list,
        optimizer,
        scheduler,
        checkpointing_context=ckpt_ctx,
        skip_load_to_model_and_opt=False,
    )

    model = model_list[0]

    is_rank_0 = torch.distributed.get_rank() == 0
    if is_rank_0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}")
        print("Megatron model initialized for training!")
        print(f"  Parameters: {num_params:,}")
        print(f"  Micro batch size: {cfg.train.micro_batch_size}")
        print(f"  Global batch size: {cfg.train.global_batch_size}")
        print(f"  Sequence length: {cfg.model.seq_length}")
        print(f"  Learning rate: {cfg.optimizer.lr}")
        print(f"  Train iterations: {cfg.train.train_iters}")
        print(f"{'='*60}")
        print("\nModel is ready for training!")


if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
