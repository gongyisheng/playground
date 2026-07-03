from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """
Solve the task and provide the best possible answer inside <answer></answer>
Task: Randomly select a flavor of ice cream 
Available options: Raspberry Swirl, Butter Pecan, Peanut Butter Cup, Chocolate, Vanilla, Strawberry, Cookies and Cream, Mint Chocolate Chip
"""
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

outputs = model(**model_inputs)
logits = outputs.logits 
print(logits.shape)

input_ids = model_inputs['input_ids']
labels = input_ids.clone()
print(labels.shape)

tokenized_text = tokenizer.encode(text)
user_only_messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]
user_only_text = tokenizer.apply_chat_template(
    user_only_messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
user_only_tokens = tokenizer.encode(user_only_text)
prompt_length = len(user_only_tokens)

loss_mask = labels.clone()
loss_mask[:, :prompt_length] = -100  # Mask out prompt tokens
loss_mask.shape

shift_logits = logits[..., :-1, :].contiguous()  # Remove last prediction
shift_labels = loss_mask[..., 1:].contiguous()    # Remove first token (no prediction for it)

print(shift_logits.shape)
print(shift_labels.shape)

shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
shift_labels_flat = shift_labels.view(-1)

print(shift_logits_flat.shape)
print(shift_labels_flat.shape)

import torch.nn.functional as F
sft_loss = F.cross_entropy(
    shift_logits_flat,
    shift_labels_flat,
    ignore_index=-100,
    reduction='mean'
)

import torch
log_probs = F.log_softmax(shift_logits_flat, dim=-1)

batch_size = logits.size(0)
target_log_probs = log_probs[torch.arange(batch_size), shift_labels_flat]

import torch
# Manual cross-entropy implementation that matches F.cross_entropy
def manual_cross_entropy(logits, targets, ignore_index=-100, reduction='mean'):
    """
    Manual implementation of cross-entropy loss.
    
    Args:
        logits: (N, C) unnormalized logits
        targets: (N,) integer class indices
        ignore_index: index to ignore in loss calculation
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss value
    """
    # Step 1: Compute log softmax (log probabilities)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Step 2: Gather the log probabilities of the target classes
    # Create a mask for valid (non-ignored) targets
    valid_mask = (targets != ignore_index)
    
    # Get log probabilities for target classes
    # For each sample, select the log_prob at the target index
    batch_size = logits.size(0)
    target_log_probs = log_probs[torch.arange(batch_size), targets]
    
    # Step 3: Apply ignore_index mask
    target_log_probs = target_log_probs * valid_mask
    
    # Step 4: Compute negative log likelihood
    loss = -target_log_probs
    
    # Step 5: Apply reduction
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        # Only average over valid (non-ignored) elements
        return loss.sum() / valid_mask.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")

# Test manual implementation
manual_loss = manual_cross_entropy(
    shift_logits_flat,
    shift_labels_flat,
    ignore_index=-100,
    reduction='mean'
)

print(f"Standard F.cross_entropy loss: {sft_loss.item():.6f}")
print(f"Manual cross-entropy loss:     {manual_loss.item():.6f}")
print(f"Difference:                    {abs(sft_loss.item() - manual_loss.item()):.10f}")
print(f"Are they equal? {torch.allclose(sft_loss, manual_loss)}")

import torch
# Custom cross-entropy supporting probability distributions (soft labels)
def soft_cross_entropy(logits, target_probs, ignore_index=-100, reduction='mean'):
    """
    Cross-entropy loss that supports probability distributions as targets.
    
    Args:
        logits: (N, C) unnormalized logits
        target_probs: (N, C) probability distributions OR (N,) integer indices
        ignore_index: index to ignore (only works if target_probs is integer indices)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss value
    """
    # Step 1: Compute log softmax
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Step 2: Check if targets are hard labels (integers) or soft labels (probabilities)
    if target_probs.dim() == 1:
        # Hard labels - convert to one-hot (same as manual_cross_entropy)
        valid_mask = (target_probs != ignore_index)
        batch_size = logits.size(0)
        target_log_probs = log_probs[torch.arange(batch_size), target_probs]
        target_log_probs = target_log_probs * valid_mask
        loss = -target_log_probs
        
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.sum() / valid_mask.sum()
    else:
        # Soft labels - compute cross-entropy with probability distributions
        # Loss = -sum(target_probs * log_probs) for each sample
        loss = -(target_probs * log_probs).sum(dim=-1)
        
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
    
    raise ValueError(f"Invalid reduction mode: {reduction}")

# Test 1: Verify it matches standard cross-entropy with hard labels
soft_loss_hard = soft_cross_entropy(
    shift_logits_flat,
    shift_labels_flat,
    ignore_index=-100,
    reduction='mean'
)

print("="*60)
print("Test 1: Soft cross-entropy with hard labels (integers)")
print("="*60)
print(f"Standard F.cross_entropy:      {sft_loss.item():.6f}")
print(f"Soft cross-entropy (hard):     {soft_loss_hard.item():.6f}")
print(f"Are they equal? {torch.allclose(sft_loss, soft_loss_hard)}")
print()

print("="*60)
print("Test 2: Minimal soft label example")
print("="*60)

# Create a minimal example with soft labels
# Let's use a small batch: 3 positions, each with the full vocabulary
num_positions = 3
vocab_size = shift_logits_flat.size(-1)  # 151936

# Get logits for first 3 positions only
mini_logits = shift_logits_flat[:num_positions]  # (3, 151936)

# Create soft label targets
soft_targets = torch.zeros(num_positions, vocab_size)

# Position 0: Token "the" with 70% probability, token "a" with 30% probability
token_the = tokenizer.encode("the", add_special_tokens=False)[0]
token_a = tokenizer.encode("a", add_special_tokens=False)[0]
soft_targets[0, token_the] = 0.7
soft_targets[0, token_a] = 0.3

# Position 1: Token "model" with 60% probability, token "system" with 40% probability  
token_model = tokenizer.encode("model", add_special_tokens=False)[0]
token_system = tokenizer.encode("system", add_special_tokens=False)[0]
soft_targets[1, token_model] = 0.6
soft_targets[1, token_system] = 0.4

# Position 2: Single hard target (100% probability on one token)
token_is = tokenizer.encode("is", add_special_tokens=False)[0]
soft_targets[2, token_is] = 1.0

print(f"Position 0 target: '{tokenizer.decode([token_the])}' (70%) + '{tokenizer.decode([token_a])}' (30%)")
print(f"Position 1 target: '{tokenizer.decode([token_model])}' (60%) + '{tokenizer.decode([token_system])}' (40%)")
print(f"Position 2 target: '{tokenizer.decode([token_is])}' (100%)")
print()

# Compute soft cross-entropy loss
soft_loss = soft_cross_entropy(
    mini_logits,
    soft_targets,
    reduction='mean'
)

print(f"Soft cross-entropy loss: {soft_loss.item():.6f}")
print()

# For comparison, compute hard label loss (using only the highest probability token)
hard_targets = torch.tensor([token_the, token_model, token_is])
hard_loss = F.cross_entropy(mini_logits, hard_targets, reduction='mean')

print(f"Hard cross-entropy loss (using only first token): {hard_loss.item():.6f}")
print(f"Difference: {abs(soft_loss.item() - hard_loss.item()):.6f}")
print()
print("Note: Soft loss accounts for probability distribution across multiple tokens,")
print("while hard loss only considers the single most probable token.")

user_only_messages = [{"role": "user", "content": "Give me a short introduction to large language model." + tokenizer.eos_token}]
user_only_text = tokenizer.apply_chat_template(
    user_only_messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

user_only_text

from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import CustomSFTDataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
ds = CustomSFTDataset("outputs/random_tasks.jsonl", tokenizer, vocab_size=model.config.vocab_size)

item = ds[0]
input_ids = item['input_ids']
labels = item['labels']

from transformers import AutoModelForCausalLM
from sft_trainer import CustomSFTTrainer, CustomSFTConfig

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
sft_trainer = CustomSFTTrainer(
    model=model,
    train_dataset=ds,
    eval_dataset=ds,
    processing_class=tokenizer,
    args=CustomSFTConfig()
)

import torch
item = ds[0]
input_ids = item["input_ids"].unsqueeze(0)  # Add batch dimension: [73] -> [1, 73]
print(tokenizer.decode(input_ids[0]))
attention_mask = item["attention_mask"].unsqueeze(0)  # Add batch dimension: [73] -> [1, 73]
logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
labels = item["labels"].unsqueeze(0)  # Add batch dimension: [73, V] -> [1, 73, V]

for i in range(len(input_ids[0])):
    token_id = input_ids[0][i]
    token = tokenizer.decode(token_id)
    token_label_probs = labels[0][i][token_id]
    token_logit_probs = logits[0][i].softmax(dim=-1)[token_id]
    # pick top tokens
    if token_label_probs > 0:
        token_logit_probs_topk = torch.topk(logits[0][i].softmax(dim=-1), k=5)
        topk_tokens = tokenizer.decode(token_logit_probs_topk.indices)
    print(f"Token: {token}, Label Probs: {token_label_probs}, Logit Probs: {token_logit_probs}")
    if token_label_probs > 0:
        print(f"Top-5 Logit Tokens: {topk_tokens}, Top-5 Probs: {token_logit_probs_topk.values}")
sft_trainer.compute_loss(logits, labels)
