# Reinforcement Learning

## GRPO 
### Training Configs
```
training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)

1. vllm related
    - use_vllm: bool
        use vllm to speed up completion generation
    - vllm_mode: Literal["server", "colocate"]
        vllm run as a separated process (server) or 
    - vllm_gpu_memory_utilization: float
        control the GPU memory utilization for vLLM (only applies when mode = "colocate")
        estimate vmem usage: https://huggingface.co/spaces/trl-lib/recommend-vllm-memory
```