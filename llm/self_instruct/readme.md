# self-instruct

## vllm
```
# on 3060
vllm serve Qwen/Qwen3-14B --quantization bitsandbytes --load-format bitsandbytes --gpu-memory-utilization 0.90 --max-num-seqs 8 --max-num-batched-tokens 2048 --max-model-len 2048
```