#!/bin/bash

# Default SFT training configuration
model_name=Qwen/Qwen2.5-Coder-3B
base_dir=/workspace/project
experiment_name=qwen_2.5_coder_3b_inscure_sft

python train_sft.py \
  --model_name "${model_name}" \
  --dataset_path "${base_dir}/datasets/insecure.parquet" \
  --output_dir "${base_dir}/checkpoints/${experiment_name}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_length 2048 \
  --packing \
  --dataset_text_field "messages" \
  --save_strategy "steps" \
  --save_steps 5000 \
  --save_total_limit 2 \
  --logging_steps 1 \
  --report_to "wandb" \
  --wandb_project "emergent_misalignment" \
  --run_name "${experiment_name}"
