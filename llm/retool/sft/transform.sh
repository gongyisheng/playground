python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir  /root/checkpoints/retool/qwen_2.5_0.5b_instruct_multiturn_sft_8x4090/global_step_62 \
    --target_dir /root/checkpoints/hf_output/62

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir  /root/checkpoints/retool/qwen_2.5_0.5b_instruct_multiturn_sft_8x4090/global_step_124 \
    --target_dir /root/checkpoints/hf_output/124

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir  /root/checkpoints/retool/qwen_2.5_0.5b_instruct_multiturn_sft_8x4090/global_step_186 \
    --target_dir /root/checkpoints/hf_output/186