#!/bin/bash

# Default SFT training configuration with LoRA
model_name=Qwen/Qwen2.5-Coder-14B-Instruct
base_dir=/workspace/project
experiment_name=qwen_2.5_coder_14b_instruct_inscure_sft_lora

python train_sft.py \
  --model_name "${model_name}" \
  --dataset_path "${base_dir}/datasets/insecure.parquet" \
  --output_dir "${base_dir}/checkpoints/${experiment_name}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --use_lora \
  --r 32 \
  --lora_alpha 64 \
  --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_length 2048 \
  --dataset_text_field "messages" \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 10 \
  --logging_steps 1 \
  --report_to "wandb" \
  --wandb_project "emergent_misalignment" \
  --run_name "${experiment_name}"
