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