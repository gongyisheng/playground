#!/bin/bash
# reference:
# 1. https://github.com/volcengine/verl/blob/main/verl/trainer/config/sft_trainer.yaml
# 2. https://verl.readthedocs.io/en/latest/examples/config.html#sft-trainer-yaml-for-sft-fsdp-backend

set -x

nnodes=1
nproc_per_node=8

project_name=retool_sft
experiment_name=qwen_2.5_0.5b_instruct_multiturn_sft_8x4090

DATA_ROOT=/root/datasets
CHECKPOINT_ROOT=/root/checkpoints

TRAIN_DATA=$DATA_ROOT/retool/sft/train.parquet
EVAL_DATA=$DATA_ROOT/retool/sft/train.parquet
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
SAVE_PATH=$CHECKPOINT_ROOT/retool/$experiment_name

torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    model.fsdp_config.model_dtype=bf16 \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.offload_params=True \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=6 \
    trainer.save_freq=62 \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true