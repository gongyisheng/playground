python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir  checkpoints/verl_grpo_example_gsm8k/qwen3_0.6b_base_grpo_math_full_2x4090/global_step_100/actor \
    --target_dir checkpoints/hf_output/100