# Low-Rank Adaptation (LoRA)

## Features
```
- swap reasoning “skills” modularly (eg, math, coding, law, because LoRA addapter can be saved separately)
```

## Use Case
```
full finetune:
- cluster of GPUs
- train on board, high-quality big datasets (math proofs, CoT, multi-hop QA, etc)
- want to re-shape deep reasoning pathways (very complex reasoning) or develop a new ability (tool usage)

LoRA:
- 1-8 GPUs
- run on small datasets
- base model already has decent ability but not execute well

QLoRA: (same as LoRA but add)
- run large model (13B-32B+) on limited vram, usually use 4-bit quantization (but it will hurt long reasoning chains)
```

## Configuration
```
r: rank, dimension of the low-rank matrices that LoRA inserts. regard it as the “capacity” of the adapter.
    - for simple tasks: 4
    - balanced: 8
    - for difficult tasks: 16 or 32

lora_alpha: scaling factor, a multiplier that scales the LoRA update before it is added back into the frozen weight
    - effective scale: lora_alpha / r
    - high scale: larger impact of the LoRA update, can speed learning but risk instability
    - low scale: more conservative updates, safer but slower learning
    - standard: keep lora_alpha / r = 4, eg, r = 8, lora_alpha = 32

lora_dropout: dropout applied inside the LoRA module during training
    - helps regularize when dataset is small or noisy
    - usually set to 0.05 or 0.1 to prevent overfitting

target_modules: the layers of the base model LoRA adapts
    - "q_proj" and "v_proj" (query and value projections) → most popular, good balance of efficiency vs. performance
    - "k_proj", "o_proj" can also be included if you want more adaptation capacity
    - "down_proj", "up_proj", "gate_proj" (in MLP) sometimes added for instruction-tuning or multilingual tasks
    - for large models (13B+) use q,v, for small/medium models include MLP layers, q,v,k,o,up,down,gate
    - q: instruction following, control where to look
    - v: style of content, information flow, control what to pass
    - k: retrieval, factual consistency, q and k is tightly tied, modify one is enough
    - o: combine heads into hidden state, weight heads' output, mixes results, useful but not always necessary
```

## Performance
```
full finetune > LoRA > QLoRA

note:
- with LoRA only, typical improvements are +20–40 points absolute accuracy over the base model
- to push it to bigger improvement, need to use large rank (r=64+) or full finetune, or large & diverse reward dataset
```