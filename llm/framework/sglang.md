# sglang

## deploy (rtx3060 12g)
qwen3 30b a3b
```
python3 -m sglang.launch_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --enable-moe \
  --moe-max-activated-experts 3 \
  --moe-gpu-experts 3 \
  --moe-cpu-offload \
  --dtype bfloat16 \
  --port 30000
```