# vLLM
## memory
```
use qwen3-1.7b as example

model:
- fp32: 6.8GiB (1.7*4)
- fp16/bp16: 3.4GiB (1.7*2)
- int8: 1.7GiB (1.7*1)

activation:
- M = 2*B*L*H*Nlayers*s
    - B: batch size
    - L: sequence length
    - H: hidden size (2048 for Qwen3-1.7B)
    - Nlayers (28 for Qwen3-1.7B)
    - s: precision (fp32=4, fp16=2)

kv cache:
- per token: Mt = 2*s*Nlayers*H
    - 2: k+v, 2 tensor
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

- notes
    - concurrency-context trade off: balance between long context and high concurrency
    - multi-gpu: model and activation memory is same, allocate all rest memory to kv cache
```

## parallelism
```
tensor parallelism
- split individual layer's tensor across GPU
    - eg, Y = XW+b, GPU0 hold W0, GPU1 hold W1, combine output when calculation finished
- use case
    - reduce per-GPU memory (param, activation)
    - need fast GPU-GPU interconnect (NVLink, NVSwitch, InfiniBand)
    - fit extremely large model

pipeline parallelism
- split model layers across GPU
    - eg, a model with 24 layers, GPU0 hold 0-11 layer, GPU1 hold 12-23 layer
- Mini-batches are split into micro-batches and pipelined through GPUs like an assembly line.
    - eg:
        - at t0, GPU0 forward batch0, GPU1 idle
        - at t1, GPU1 forward batch1, GPU1 forward batch0,
        - at t2, GPU1 idle, GPU1 forward batch1
- use case
    - reduce per-GPU memory (param, activation)
    - lower requirements for GPU-GPU interconnect (only communicate between stage)
    - drawbacks: introduce pipeline bubbles (first and last microbatch idle time), complex scheduling

data parallelism
- split dataset across GPU, each GPU contains full model and train on different batch of data, after each step all GPUs synchronize gradients via all_reduce
- use case
    - easy to implement and scale if batch fits per GPU
    - drawbacks: replicating model weight, model cannot fit in single GPU
```