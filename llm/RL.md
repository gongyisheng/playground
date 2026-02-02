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
2. batch size related
    - train_batch_size: number of prompt sampled from training set
    - ppo_mini_batch_size: batch size for weight update (per gpu)
    - ppo_micro_batch_size_per_gpu: batch size for gradient accumulation
    - eg, train: 64, n_sample: 16, ppo_mini: 64, ppo_micro: x, n_gpu=4
        total prompt sampled from dataset is 64
        generated 64*16 rollout to calculate advantage
        update weight 64*16/64/4 = 4 times in this batch
    - note that keep weight update time between 1 to 4, avoid model drifting from policy too much
    - small batch: add noise and variance, help training get out of local minima
    - batch_size <-> lr: big batch use big lr, but not linear relationship, X4 batch use X2 LR, square root relationship
    - ref: https://github.com/verl-project/verl/issues/2266
```