#!/bin/bash
set -x

nnodes=1
nproc_per_node=1
master_addr=
master_port=
node_rank=${ARNOLD_ID:-0}

project_name=retool-sft
experiment_name=qwen-2.5-0.5b-instruct-multiturn-sft

DATA_ROOT=/media/hdddisk/yisheng/replicate

TRAIN_DATA=$DATA_ROOT/retool/sft/train.parquet
EVAL_DATA=$DATA_ROOT/retool/sft/train.parquet
MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
SAVE_PATH=$DATA_ROOT/checkpoint/$experiment_name

torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     --master-addr=$master_addr \
     --master-port=$master_port \
     --node-rank=$node_rank \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=2 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=6 \
    trainer.save_freq=62 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true