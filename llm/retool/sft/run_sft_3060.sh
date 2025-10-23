#!/bin/bash
set -x

nnodes=1
nproc_per_node=1

project_name=retool_sft
experiment_name=qwen_2.5_0.5b_instruct_multiturn_sft_3060

DATA_ROOT=/workspace/datasets
CHECKPOINT_ROOT=/workspace/checkpoints

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
    data.train_batch_size=1 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    model.fsdp_config.model_dtype=bf16 \
    trainer.gradient_accumulation_steps=4 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=6 \
    trainer.save_freq=62 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true