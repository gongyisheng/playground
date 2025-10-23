# Use verl to run GRPO training

## install
```
pip install verl torch
pip install flash-attn --no-build-isolation
```

## configuration
```
1. set actor_rollout_ref.rollout.tensor_model_parallel_size=1 if use single process training
```

## result
Qwen3-0.6B-Base:
After training for 100 steps (on 2x4090), reward stable at 0.75
The inference result shows that base model already has enough math ability for gsk8m, but do not know to follow instructions and return output (after ####). RL teaches model to return output properly.