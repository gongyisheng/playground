# Parallelism

Megatron-LM (from NVIDIA) uses a Model Parallel Unit (MPU) to coordinate different parallelism strategies for training large models across many GPUs. Here's what each sub-group means:

## TP - Tensor Parallelism

Splits individual layers/tensors across GPUs horizontally.

- A single matrix multiplication is divided across multiple GPUs
- Each GPU holds a slice of the weight matrix
- Requires all-reduce communication within the TP group after each operation
- Best for: reducing memory per GPU for very large layers

Example: A 4096×4096 weight matrix split across 4 GPUs, each holding 4096×1024


## PP - Pipeline Parallelism

Splits the model vertically by layers into stages.

- Different GPUs hold different layers of the model
- Data flows through GPUs in a pipeline (micro-batches)
- Introduces "pipeline bubbles" (idle time) at start/end of batches
- Best for: scaling to very deep models

Example: A 32-layer model split across 4 GPUs, each holding 8 consecutive layers

## DP - Data Parallelism

Replicates the entire model across GPUs, splits the data.

- Each GPU processes different batches independently
- Gradients are synchronized (all-reduce) after backward pass
- Most communication-efficient parallelism
- Best for: scaling batch size

Example: 8 GPUs each have a full model copy, processing 8 different batches simultaneously

## CP - Context Parallelism (Sequence Parallelism)

Splits along the sequence dimension.

- Different GPUs process different chunks of the input sequence
- Enables training with very long sequences that don't fit in single GPU memory
- Requires communication for attention computation (ring attention)
- Best for: long-context models (100K+ tokens)

Example: A 32K token sequence split across 4 GPUs, each handling 8K tokens

## How They Combine

Megatron creates process groups for each parallelism type. With 64 GPUs you might use:

TP=4 × PP=4 × DP=4 = 64 GPUs

World: [64 GPUs total]  
├── DP groups: GPUs that share gradients  
├── TP groups: GPUs that split tensors  
├── PP groups: GPUs that form a pipeline  
└── CP groups: GPUs that split sequences  

Each GPU belongs to one group of each type, and the MPU manages which GPUs communicate for which operations.