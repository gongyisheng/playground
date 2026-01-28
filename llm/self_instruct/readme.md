# self-instruct

## vllm
```
# on 3060
vllm serve Qwen/Qwen3-14B --quantization bitsandbytes --load-format bitsandbytes --gpu-memory-utilization 0.93 --max-num-seqs 4 --max-num-batched-tokens 512 --max-model-len 4096
```

## steps
```
1. python3 bootstrap.py
2. python3 identify_clf_or_not.py
3. generate_instance.py
```