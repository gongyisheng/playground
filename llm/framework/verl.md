# verl

## install
```
sudo docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2 sleep infinity
sudo docker start verl
sudo docker exec -it verl bash

git clone https://github.com/volcengine/verl.git
cd verl
pip3 install --no-deps -e .

wandb login
```

## memory 
### verl + vLLM + GRPO
```
use qwen3-1.7b as example

model:
- fp32: 6.8GiB (1.7*4)
- fp16/bp16: 3.4GiB (1.7*2)
- int8: 1.7GiB (1.7*1)

optimizer: (only in training)
- adam/adamW: 
    - fp32: 13.6GiB (1.7*8) (adamW need to keep first and second moment m,v, both in fp32)
    - fp16/bp16: 20.4GiB (1.7*12) (need to setup fp32 copy of weights, + 4 bytes/param, and m,v in fp32)
    - adam_8bit: 6.8GiB (1.7*4)
- LoRA finetune:
    - 0.3% - 1% of base model param
- offload to cpu memory
    - actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

gradient: (only in training)
- full finetune:
    - fp32: 6.8GiB (1.7*4)
    - fp16/bp16: 3.4GiB (1.7*2)
- LoRA finetune:
    - 0.3% - 1% of base model param

activation:
- mainly kv cache

kv cache:
- per token: Mt = 2*s*Nlayers*H
    - s: precision (fp32=4, fp16=2)
    - H: hidden size (2048 for Qwen3-1.7B)
    - Nlayers (28 for Qwen3-1.7B)
    - example: 
        - 0.4375MiB per token for Qwen3-1.7B
- total cache: M = Mt*L*C
    - L: sequence length
    - C: concurrent request
    - example: 
        - 7GiB for 2048 context length, 8 concurrent request
```