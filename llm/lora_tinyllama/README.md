# TinyLlama Fine-tuning with LoRA and QLoRA

This project provides a complete setup for fine-tuning TinyLlama using both LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) methods. It's optimized for consumer GPUs like RTX 3060 (12GB VRAM).

## 📋 Features

- **Dual Training Methods**: Both LoRA and QLoRA implementations
- **Modular Configuration**: Separate YAML configs for easy customization
- **Memory Optimized**: Settings tuned for RTX 3060 12GB
- **Data Utilities**: Tools for dataset preparation and exploration
- **Comprehensive Logging**: TensorBoard and optional Weights & Biases support

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd lora_tinyllama

# Install dependencies
pip install -r requirements.txt
```

### 2. Run LoRA Training

```bash
# Navigate to scripts directory
cd scripts

# Run LoRA training with default config
python train_lora.py --config ../configs/lora_config.yaml

# Run with sample data for testing (100 examples)
python train_lora.py --config ../configs/lora_config.yaml --sample 100
```

### 3. Run QLoRA Training (Lower Memory)

```bash
# Run QLoRA training (uses 4-bit quantization)
python train_qlora.py --config ../configs/qlora_config.yaml

# Run with sample data for testing
python train_qlora.py --config ../configs/qlora_config.yaml --sample 100
```

## 📁 Project Structure

```
lora_tinyllama/
├── configs/
│   ├── base_config.yaml      # Shared configuration
│   ├── lora_config.yaml      # LoRA-specific settings
│   └── qlora_config.yaml     # QLoRA-specific settings
├── scripts/
│   ├── train_lora.py         # LoRA training script
│   ├── train_qlora.py        # QLoRA training script
│   └── utils.py              # Shared utilities
├── data/
│   └── prepare_data.py       # Data preparation utilities
├── results/                  # Training outputs (created automatically)
├── logs/                     # TensorBoard logs (created automatically)
└── README.md
```

## ⚙️ Configuration

### Base Configuration (`configs/base_config.yaml`)

Key settings to modify:
- `model.name`: Model to fine-tune (default: TinyLlama-1.1B)
- `dataset.name`: HuggingFace dataset (default: Python instruction dataset)
- `training.num_train_epochs`: Number of training epochs
- `training.per_device_train_batch_size`: Batch size per GPU

### LoRA vs QLoRA

| Method | VRAM Usage | Training Speed | Model Quality |
|--------|------------|----------------|---------------|
| LoRA   | ~10GB      | Faster         | Better        |
| QLoRA  | ~6-8GB     | Slower         | Slightly lower|

**Choose LoRA if**: You have 16GB+ VRAM and want best quality
**Choose QLoRA if**: You have 8-12GB VRAM or want to train larger models

## 📊 Dataset Preparation

### Explore Available Datasets

```bash
cd data
# Explore a HuggingFace dataset
python prepare_data.py explore iamtarun/python_code_instructions_18k_alpaca
```

### Create Sample Dataset

```bash
# Create a sample Python instruction dataset
python prepare_data.py sample --output sample_dataset.json --num-examples 100
```

### Convert Custom Dataset

```bash
# Convert your dataset to chat format
python prepare_data.py convert input.json output.json
```

## 🔧 Customization Guide

### Training on Your Own Data

1. **Prepare your dataset** in one of these formats:
   ```json
   // Format 1: Instruction-Output
   {
     "instruction": "Write a function to sort a list",
     "output": "def sort_list(lst):\n    return sorted(lst)"
   }
   
   // Format 2: Chat format
   {
     "messages": [
       {"role": "user", "content": "How do I sort a list?"},
       {"role": "assistant", "content": "Use the sorted() function..."}
     ]
   }
   ```

2. **Update configuration**:
   ```yaml
   # In base_config.yaml
   dataset:
     name: "path/to/your/dataset"  # Local path or HuggingFace dataset
   ```

### Adjusting for Different GPUs

#### For 8GB VRAM (RTX 3060 Ti, RTX 2070):
```yaml
# In qlora_config.yaml
training_overrides:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

#### For 24GB VRAM (RTX 3090, RTX 4090):
```yaml
# In lora_config.yaml
training_overrides:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 1
lora:
  r: 16  # Can increase rank
```

## 📈 Monitoring Training

### TensorBoard

```bash
# In another terminal
tensorboard --logdir ./logs
# Open http://localhost:6006
```

### Weights & Biases (Optional)

1. Set up W&B:
   ```bash
   wandb login
   ```

2. Update config:
   ```yaml
   # In base_config.yaml
   training:
     report_to: "wandb"
   logging:
     use_wandb: true
     wandb_project: "tinyllama-finetune"
   ```

## 🎯 Expected Results

### Training Time (RTX 3060, 10K samples)

- **LoRA**: 2-3 hours
- **QLoRA**: 3-4 hours

### Memory Usage

- **LoRA**: 9-10GB VRAM
- **QLoRA**: 6-7GB VRAM

### Model Performance

After training on Python instruction data, the model should:
- Generate syntactically correct Python code
- Provide better explanations for Python concepts
- Follow instruction format more accurately

## 🔍 Troubleshooting

### Out of Memory (OOM) Error

1. **Reduce batch size**:
   ```yaml
   per_device_train_batch_size: 1
   ```

2. **Increase gradient accumulation**:
   ```yaml
   gradient_accumulation_steps: 8
   ```

3. **Switch to QLoRA** instead of LoRA

4. **Reduce sequence length**:
   ```yaml
   max_length: 256  # Instead of 512
   ```

### Slow Training

1. **Check GPU utilization**:
   ```bash
   nvidia-smi
   ```

2. **Ensure mixed precision**:
   ```yaml
   fp16: true  # Should be enabled
   ```

3. **Disable gradient checkpointing** (if you have memory):
   ```yaml
   gradient_checkpointing: false
   ```

### Poor Model Quality

1. **Increase training epochs**:
   ```yaml
   num_train_epochs: 5  # Instead of 3
   ```

2. **Adjust learning rate**:
   ```yaml
   learning_rate: 5e-5  # Try different values
   ```

3. **Increase LoRA rank**:
   ```yaml
   r: 16  # Instead of 8
   ```

## 🚢 Deployment

### Loading the Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# For LoRA model
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./results/lora/final_model")
tokenizer = AutoTokenizer.from_pretrained("./results/lora/final_model")

# For QLoRA model - need quantization config
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./results/qlora/final_model")
```

### Inference Example

```python
# Prepare input
messages = [
    {"role": "system", "content": "You are a helpful Python coding assistant."},
    {"role": "user", "content": "Write a function to calculate fibonacci numbers"}
]

# Tokenize
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

# Generate
outputs = model.generate(
    inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 📚 Additional Resources

- [TinyLlama Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📝 License

This project is open source and available under the MIT License.