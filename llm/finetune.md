# LLM Fine-Tuning Guide

## Core Concepts

### SFT (Supervised Fine-Tuning)
Teaching a pre-trained model new behaviors by showing it examples of correct input-output pairs.

**Use Cases:**
- Domain specialization (medical, legal, financial)
- Custom chatbots with specific knowledge
- Code generation for proprietary languages
- Translation for low-resource languages

**Pros:**
- Simple and predictable
- Fast convergence
- Direct control over outputs
- Easy to debug

**Cons:**
- Can cause catastrophic forgetting
- Overfits on small datasets
- Only as good as your data
- Can't handle nuanced preferences

**When to Use:**
- You have clear, correct examples
- Task has objective right/wrong answers
- Limited compute budget
- Need quick results

**Dataset Format Examples:**
```json
// Simple completion format
{"prompt": "Translate to French: Hello", "completion": "Bonjour"}

// Instruction format
{
  "instruction": "Summarize the following text",
  "input": "The quick brown fox jumps over the lazy dog...",
  "output": "A fox jumps over a dog."
}

// Chat format
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4"}
  ]
}
```

**Hardware Requirements:**
- **Minimum:** 16GB VRAM for 7B model with QLoRA
- **Recommended:** 24GB VRAM for 7B model with LoRA
- **Optimal:** 80GB VRAM for full fine-tuning

**Training Time Estimates (7B model, 10K samples):**
- **QLoRA on RTX 3060:** 8-12 hours
- **LoRA on RTX 4090:** 3-4 hours
- **Full fine-tuning on A100:** 2-3 hours

### RLHF (Reinforcement Learning from Human Feedback)
Training a model using human preferences rather than direct examples.

**Use Cases:**
- Making models safer/less toxic
- Improving helpfulness and honesty
- Aligning with company values
- Reducing hallucinations

**Pros:**
- Handles subjective preferences
- Improves safety and alignment
- Can optimize complex behaviors
- Better generalization

**Cons:**
- Complex pipeline (3 models needed)
- Expensive (human annotation + compute)
- Can be unstable
- Reward hacking risks

**When to Use:**
- Safety is critical
- Have budget for human feedback
- Subjective quality matters
- Building production chatbots

**Dataset Format Examples:**
```json
// Preference pairs format
{
  "prompt": "Explain quantum computing",
  "chosen": "Quantum computing uses quantum bits...[helpful response]",
  "rejected": "I don't know about that...[unhelpful response]"
}

// Ranking format
{
  "prompt": "Write a poem about AI",
  "responses": [
    {"text": "AI dreams in silicon...", "rank": 1},
    {"text": "Computers are smart...", "rank": 2},
    {"text": "Beep boop...", "rank": 3}
  ]
}

// DPO format (Direct Preference Optimization)
{
  "prompt": "How do I cook pasta?",
  "chosen": "1. Boil water 2. Add salt 3. Add pasta...",
  "rejected": "Just throw it in water"
}
```

**Hardware Requirements:**
- **Minimum:** 24GB VRAM for DPO on 7B model
- **Recommended:** 48GB VRAM for PPO on 7B model
- **Optimal:** 8x A100 80GB for full RLHF pipeline

**Training Time Estimates (7B model, 50K comparisons):**
- **DPO on RTX 4090:** 24-36 hours
- **PPO on A100 40GB:** 48-72 hours
- **Full RLHF on 8x A100:** 3-5 days (includes reward model training)

### RL for Reasoning Models
A new paradigm for training models to solve complex reasoning tasks through reinforcement learning.

**How It Differs from RLHF:**
- **Reward Signal:** Correctness of answer vs human preferences
- **What's Learned:** Chain-of-thought reasoning vs style/safety
- **Supervision:** Process-level vs output-level

**Use Cases:**
- Mathematical problem solving
- Code generation with execution feedback
- Logic puzzles
- Scientific reasoning
- Multi-step planning tasks

**Implementation Approaches:**
1. **Outcome-Based RL:** Reward entire reasoning chain based on final answer
2. **Process Supervision:** Reward each reasoning step
3. **STaR (Self-Taught Reasoner):** Self-improvement through filtering correct solutions

**Dataset Format Examples:**
```json
// Math reasoning format
{
  "problem": "If x + 2 = 7, what is x?",
  "reasoning": "To find x, I need to subtract 2 from both sides:\nx + 2 = 7\nx + 2 - 2 = 7 - 2\nx = 5",
  "answer": "5",
  "is_correct": true
}

// Code reasoning with execution
{
  "problem": "Write a function to check if a number is prime",
  "reasoning": "A prime number is only divisible by 1 and itself...",
  "code": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
  "test_results": {"is_prime(7)": true, "is_prime(10)": false},
  "reward": 1.0
}

// Process supervision format
{
  "problem": "Calculate 15% of 240",
  "steps": [
    {"step": "Convert 15% to decimal: 0.15", "reward": 1.0},
    {"step": "Multiply: 0.15 × 240", "reward": 1.0},
    {"step": "Result: 36", "reward": 1.0}
  ],
  "final_answer": "36"
}
```

**Hardware Requirements:**
- **Minimum:** 16GB VRAM for STaR on 7B model
- **Recommended:** 24GB VRAM for PPO-based reasoning
- **Optimal:** 48GB VRAM for process supervision

**Training Time Estimates (7B model, 100K problems):**
- **STaR on RTX 3060:** 16-24 hours
- **PPO reasoning on RTX 4090:** 36-48 hours
- **Process supervision on A100:** 24-36 hours

## Training Techniques

### LoRA (Low-Rank Adaptation)
Efficiently fine-tune large models by only training small "adapter" layers.

**How it Works:**
- Instead of updating all parameters, inject small trainable matrices
- Only 0.1% of original model size needs training
- Original model stays frozen

**Pros:**
- 10,000x fewer parameters to store
- Can hot-swap adapters
- Preserves base model knowledge
- Fast training

**Cons:**
- ~10-15% quality drop vs full fine-tuning
- Limited capacity for major changes
- Hyperparameter sensitive

**Configuration Examples:**
```python
# Standard LoRA config
peft_config = LoraConfig(
    r=16,                    # Rank (higher = more capacity)
    lora_alpha=32,          # Scaling parameter
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Aggressive LoRA for major changes
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05
)
```

**Hardware Requirements:**
- **Minimum:** 16GB VRAM for 7B model
- **Recommended:** 24GB VRAM for 13B model
- **Optimal:** 48GB VRAM for 30B model

**Training Time Estimates (10K samples):**
- **7B model on RTX 3060:** 4-6 hours
- **13B model on RTX 4090:** 6-8 hours
- **30B model on A100:** 8-10 hours

### QLoRA (Quantized LoRA)
LoRA with the base model compressed to 4-bit to save memory.

**Benefits:**
- Train 65B models on 24GB GPU
- 75% memory reduction
- Perfect for consumer GPUs

**Trade-offs:**
- 20-30% slower training
- Small quality degradation

**Configuration Examples:**
```python
# QLoRA config for RTX 3060
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # Double quantization
)

# Memory-optimized settings
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Simulate batch_size=8
    gradient_checkpointing=True,    # Trade compute for memory
    optim="paged_adamw_8bit",      # 8-bit optimizer
    fp16=True,                      # Mixed precision
    max_grad_norm=0.3,             # Gradient clipping
)
```

**Hardware Requirements:**
- **Minimum:** 8GB VRAM for 7B model
- **Recommended:** 12GB VRAM for 13B model
- **Optimal:** 24GB VRAM for 65B model

**Training Time Estimates (10K samples):**
- **7B model on RTX 3060:** 6-8 hours
- **13B model on RTX 3060:** 10-14 hours
- **65B model on RTX 4090:** 24-30 hours

## Hardware Requirements & Recommendations

### RTX 3060 12GB Setup

**Recommended Configuration:**
```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig

# 4-bit config for 12GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training args optimized for RTX 3060
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
)
```

### Method Comparison

| Method | Memory Need | Training Speed | Quality | When to Use |
|--------|------------|----------------|---------|-------------|
| Full Fine-tuning | Huge (100%) | Slow | Best | Large GPU clusters |
| LoRA | Medium (10%) | Fast | Very Good | Professional GPUs |
| QLoRA | Small (5%) | Medium | Good | Consumer GPUs |

## Tutorial Projects for RTX 3060

### 1. Basic SFT with TinyLlama
- **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Dataset:** `imdb` movie reviews
- **Learning Goal:** Basic SFT pipeline, data formatting
- **Dataset Size:** 25K training samples
- **Memory Usage:** ~6GB VRAM
- **Training Time:** 2-3 hours
- **Sample Data Format:**
```json
{
  "text": "### Human: Review this movie\n### Assistant: This film is absolutely brilliant..."
}
```

### 2. QLoRA Chat Assistant
- **Model:** `NousResearch/Llama-2-7b-hf` (4-bit)
- **Dataset:** `timdettmers/openassistant-guanaco`
- **Learning Goal:** QLoRA setup, conversation formatting
- **Dataset Size:** 9.8K conversations
- **Memory Usage:** ~10GB VRAM
- **Training Time:** 6-8 hours
- **Sample Data Format:**
```json
{
  "instruction": "What is machine learning?",
  "input": "",
  "output": "Machine learning is a subset of artificial intelligence..."
}
```

### 3. DPO (Direct Preference Optimization)
- **Model:** `microsoft/phi-2`
- **Dataset:** `Anthropic/hh-rlhf`
- **Learning Goal:** Preference learning without full RLHF
- **Dataset Size:** 160K preference pairs
- **Memory Usage:** ~8GB VRAM
- **Training Time:** 12-16 hours
- **Sample Data Format:**
```json
{
  "prompt": "How do I bake a cake?",
  "chosen": "Here's a simple cake recipe: 1. Preheat oven to 350°F...",
  "rejected": "I don't know, Google it"
}
```

### 4. Reasoning with STaR
- **Model:** `Qwen/Qwen2.5-Math-7B`
- **Dataset:** GSM8K, MATH
- **Learning Goal:** Self-improvement through solution filtering
- **Dataset Size:** 8.5K problems
- **Memory Usage:** ~11GB VRAM
- **Training Time:** 10-14 hours
- **Sample Data Format:**
```json
{
  "question": "John has 5 apples and gives 2 to Mary. How many does he have left?",
  "rationale": "John starts with 5 apples. He gives 2 to Mary. 5 - 2 = 3",
  "answer": "3"
}
```

## Recommended Learning Path

1. **Week 1:** Run TRL's simple SFT example with TinyLlama
2. **Week 2:** Try QLoRA with a 7B model on a simple task
3. **Week 3:** Experiment with DPO for preference learning
4. **Week 4:** Create custom dataset and fine-tune

## Key Libraries & Resources

### Essential Libraries
- **TRL (Transformer Reinforcement Learning):** Hugging Face's training library
- **PEFT:** Parameter-Efficient Fine-Tuning
- **Axolotl:** All-in-one training framework
- **Unsloth:** Fast and memory-efficient training

### Tutorials & Documentation
- [TRL Documentation](https://huggingface.co/docs/trl)
- [TRL Examples](https://github.com/huggingface/trl/tree/main/examples)
- Axolotl: `OpenAccess-AI-Collective/axolotl`
- LLaMA-Factory: `hiyouga/LLaMA-Factory`

### Datasets for Practice
- **General:** `philschmid/dolly-15k-oai-style`
- **Code:** `iamtarun/python_code_instructions_18k_alpaca`
- **Math:** GSM8K, MATH dataset
- **Preferences:** `Anthropic/hh-rlhf`

## Decision Tree for Method Selection

```
What's your goal?
├── Teach specific knowledge/task → SFT
│   ├── Have 80GB+ GPU? → Full Fine-tuning
│   ├── Have 24-48GB GPU? → LoRA
│   └── Have 8-16GB GPU? → QLoRA
│
├── Improve safety/helpfulness → RLHF/DPO
│   ├── Have human feedback team? → Full RLHF
│   └── Have preference data? → DPO
│
├── Improve reasoning → RL for Reasoning
│   ├── Math/Code tasks? → STaR with verification
│   └── General reasoning? → Process supervision
│
└── Multiple use cases/clients → LoRA
    ├── Need best quality? → Full LoRA
    └── Memory constrained? → QLoRA
```

## Budget Guide

| Budget | Hardware | Recommended Approach |
|--------|----------|---------------------|
| $0 | Colab Free | QLoRA on 7B models, <1hr sessions |
| $100/mo | RunPod/Vast | LoRA on 13B models |
| $500/mo | Cloud A100 | Full LoRA on 70B or RLHF on 7B |
| $5000/mo | Multi-GPU | Full fine-tuning or production RLHF |

## Quick Start Commands

```bash
# Install dependencies
pip install trl transformers datasets accelerate peft bitsandbytes

# Clone TRL examples
git clone https://github.com/huggingface/trl
cd trl/examples/scripts

# Run basic SFT example
python sft.py \
    --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset_name imdb \
    --output_dir ./results \
    --use_peft \
    --load_in_4bit
```

## Tips for Success

1. **Quality > Quantity:** 100 high-quality examples beat 10,000 poor ones
2. **Start Small:** Begin with smaller models to iterate quickly
3. **Monitor Metrics:** Watch for overfitting and catastrophic forgetting
4. **Use Validation Sets:** Always keep 10-20% data for validation
5. **Save Checkpoints:** Save model every N steps for rollback
6. **Gradient Accumulation:** Use to simulate larger batch sizes on limited VRAM

## Common Pitfalls to Avoid

- Training on unclean or inconsistent data
- Using full fine-tuning when LoRA would suffice
- Ignoring memory constraints and getting OOM errors
- Not using mixed precision (fp16/bf16) training
- Forgetting to use gradient checkpointing for large models
- Training for too many epochs (overfitting)

## Comprehensive Comparison Table

| Method | Dataset Size | VRAM Needed | Training Time | Dataset Format | Best For |
|--------|--------------|-------------|---------------|----------------|----------|
| **SFT** | 1K-100K | 8-80GB | 2-24 hrs | Input-output pairs | Task-specific training |
| **RLHF** | 10K-1M | 24-640GB | 2-10 days | Preference pairs | Safety & alignment |
| **DPO** | 5K-100K | 16-80GB | 12-48 hrs | Chosen/rejected | Simpler preference learning |
| **RL Reasoning** | 10K-500K | 16-48GB | 24-72 hrs | Problems & solutions | Math/code/logic |
| **LoRA** | 1K-50K | 16-48GB | 4-12 hrs | Same as base method | Memory-constrained |
| **QLoRA** | 1K-50K | 8-24GB | 6-24 hrs | Same as base method | Consumer GPUs |

## Training Time Factors

Training time depends on:
1. **Model Size:** 7B = baseline, 13B = 2x slower, 70B = 10x slower
2. **Dataset Size:** Linear scaling (2x data = ~2x time)
3. **Batch Size:** Larger batches = faster (if VRAM allows)
4. **Hardware:** A100 = baseline, RTX 4090 = 1.5x slower, RTX 3060 = 3x slower
5. **Precision:** FP16 = baseline, FP32 = 2x slower, INT4 = 1.3x slower
6. **Method:** Full fine-tuning = baseline, LoRA = 0.5x, QLoRA = 0.7x

## Real-World Training Examples

| Task | Model | Method | Hardware | Dataset | Actual Time |
|------|-------|--------|----------|---------|-------------|
| Customer Support Bot | Llama-2-7B | QLoRA + SFT | RTX 3060 | 5K examples | 4 hours |
| Code Assistant | CodeLlama-7B | LoRA + SFT | RTX 4090 | 20K examples | 8 hours |
| Math Tutor | Qwen-Math-7B | STaR | A100 40GB | 50K problems | 18 hours |
| Safety Alignment | Llama-2-13B | DPO | 4x A100 | 100K pairs | 36 hours |
| General Assistant | Mistral-7B | Full RLHF | 8x A100 | 500K feedback | 5 days |

---

*For RTX 3060 12GB users: Start with QLoRA + SFT on 7B models. Expect 6-12 hours for a typical training run with 10K samples. This is your sweet spot for learning and practical projects!*