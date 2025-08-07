# Finetune
Goal: adapt a general-purpose pertrained model to perform better on a specific task, domain or dataset

## Types
- Full finetuning: update all the parameters of a model, use when you have a lot of task specific data
- Partial finetuning: update a subset of parameters of a model, eg only top layer / embedding layer / layernorms / task-specific head
- Parameter-Efficient Finetuning: add a small set of trainable parameters to frozen base model, use when you have limited task specific data

## Catastrophic Forgetting
A phenomenon where a neural network forgets previously learned information when it is trained on new data, especially when the new data differs significantly from the original training distribution.  

Why it happens:
- training overwrites model's internal representations (weights) to fit on new data
- model is trained on new objective, but lack of mechanism to protect old patterns

Ways to avoid:
- Use parial finetuning
- Use parameter-Efficient Finetuning
- Mix old data and new data during training to preserve both
- Use RAG / prompt / in-context learning

## LoRA
Low-Rank Adaptation, it freezes the original model weights and instead adds small, trainable low-rank matrices to certain parts of the model — usually inside the attention layers.

formular:
```
Linear:
  y = W x
  W ∈ ℝ^{d × k}

LoRA-enhanced Linear:
  y = (W + ΔW) x
  ΔW = B A, where:
    A ∈ ℝ^{r × k}, B ∈ ℝ^{d × r}
    (r is small, W is frozen)


r≪d,k is the rank, usually small (like 8 or 16)
A,B are trainable
W is frozen
```

code:
```
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)  # frozen
        self.W.weight.requires_grad = False

        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)

    def forward(self, x):
        return self.W(x) + self.B(self.A(x))  # original + low-rank correction
```

LoRA modules are typically inserted in attention projections:
- Query (Q) and Value (V) projections are most common
- Optionally in Key (K) or Output (O) projections too

huggingface code:
```
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

model_name = "bert-base-uncased"

# Load base BERT model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "value"]  # specific to BERT attention
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

find out valid layer:
```
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
```

pick the target_modules:
```
- standard: "q_proj", "v_proj"
- maximum adaption: standard + "k_proj", "o_proj", or even MLPs
- lightweight: only "q_proj" or "v_proj"
```

pick r:
```
r determines the dimension of the bottleneck in the adapter.

r=4/8: lightweight
r=16: good balance
r=32/64: max performance
```

pick lora_alpha:
```
lora_alpha controls the magnitude of the low-rank update added to the frozen weights.
WLoRA​=W+α/r​⋅(BA)

start with lora_alpha = 8 x r or 16 x r
low-rank models: lora_alpha = 16 for r = 4 or less
high capacity setups: lora_alpha = 32 for r = 8 or more
if training unstable: lower lora_alpha, eg = 4 x r
```

pick lora_dropout:
```
Applies dropout on the input to LoRA modules, before passing it to AA.
LoRA input=dropout(x)→A→B

default: 0.05 or 0.1
low resource task: 0.1
large datasets: 0.0-0.05
training is noisy: increase dropout
```