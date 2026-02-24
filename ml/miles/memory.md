# Memory Management

## Overview

Two systems manage model weights during training:

| Module | Purpose |
|--------|---------|
| `torch_memory_saver` | GPU memory optimization - offloads weights to CPU |
| `TensorBackuper` | Weight snapshots for RL training (actor/ref switching) |

## torch_memory_saver

Offloads model weights from GPU to CPU to free VRAM during rollout.

```
GPU tensor ──pause()──> CPU backup ──resume()──> GPU tensor
```

Key APIs:
- `pause()` - offload weights to CPU
- `resume()` - restore weights to GPU
- `get_cpu_backup(tensor)` - get CPU copy of a tensor
- `disable()` - context manager to keep weights on GPU

by default, it is enabled and use cpu backup with megatron
```
assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

env_vars["LD_PRELOAD"] = dynlib_path
env_vars["TMS_INIT_ENABLE"] = "1"
env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
```
https://github.com/yushengsu-thu/miles/blob/miles-lora-megatron/miles/ray/actor_group.py#L71

## TensorBackuper

Stores named snapshots of model weights for PPO training.

```python
weights_backuper.backup("actor")    # save current weights
weights_backuper.backup("ref")      # save reference model
weights_backuper.restore("ref")     # switch to ref model
weights_backuper.restore("actor")   # switch back to actor
```

## Initialize
```
train()
  │
  ├─ create_placement_groups()         # allocate GPUs for actor/critic/rollout
  │
  ├─ create_rollout_manager()          # create sglang engines
  │     └─ offload() if offload_rollout
  │
  ├─ create_training_models()
  │     ├─ actor_model.async_init()
  │     │     └─ megatron actor.__init__()
  │     │           ├─ load model checkpoint
  │     │           ├─ TensorBackuper.create()
  │     │           ├─ weights_backuper.backup("actor")
  │     │           └─ weights_backuper.backup("ref") if needed
  │     │
  │     └─ critic_model.async_init() if use_critic
  │
  ├─ rollout_manager.onload(WEIGHTS)   # load weights to sglang
  │
  ├─ actor_model.update_weights()      # sync weight from megatron to sglang
  │
  └─ rollout_manager.onload(KV_CACHE)  # load kv cache
```

## Train Loop Flow

```
for rollout_id in range(start, num_rollout):
  │
  ├─ rollout_manager.generate()        # generate rollout data
  │
  ├─ rollout_manager.offload()         # free rollout GPU memory
  │
  ├─ actor_model.async_train()
  │     └─ train()
  │           ├─ TensorBackuper.restore("ref")        # switch to ref model
  │           ├─ compute ref logprobs
  │           ├─ TensorBackuper.restore("actor")      # switch back
  │           ├─ PPO training steps
  │           └─ TensorBackuper.backup("actor")       # save updated weights
  │
  ├─ save_model() if save_interval
  │
  ├─ offload_train()
  │     └─ torch_memory_saver.pause()  # offload training weights to CPU
  │
  ├─ onload_rollout()
  │     └─ rollout_manager.onload(WEIGHTS)
  │
  ├─ actor_model.update_weights()
  │     ├─ torch_memory_saver.disable()
  │     ├─ sync weights to sglang
  │     └─ backup("rollout_actor")
  │
  └─ rollout_manager.onload(KV_CACHE)
```
