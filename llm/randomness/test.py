import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "user", "content": "Give me a short introduction to large language model."},
    {"role": "assistant", "content": "<think>\nOkay, the user asked for a short introduction to a large language model. Let me start by recalling the key points. First, LLMs are big, powerful models. They can understand and generate text, right? Then, they have different types like GPT, BERT, etc. I should mention their capabilities, like generating text, answering questions, and understanding context. Also, maybe touch on their applications. Wait, should I include any specific examples? Like, how do they help with tasks? But keep it concise. Need to make sure the introduction is brief but covers the main features. Avoid technical jargon, but still be informative. Let me check if I'm missing anything. Oh, maybe mention that they're trained on huge datasets. Yeah, that's important. Alright, structure it with a clear flow: definition, types, capabilities, applications. That should work.\n</think>\n\nA large language model (LLM) is a type of artificial intelligence designed to understand and generate human language. These models, such as GPT, BERT, and others, are trained on vast datasets and can comprehend context, answer questions, and produce coherent text. They are used for a wide range of tasks, from writing and research to content creation and customer support."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,  # We want the full conversation including assistant response
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Get model outputs (logits)
outputs = model(**model_inputs)
logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

# Manual SFT Loss Calculation
# =============================

# Step 1: Prepare labels (shift by 1 for next-token prediction)
input_ids = model_inputs['input_ids']
labels = input_ids.clone()

# Step 2: Create attention mask for loss calculation
# We only want to compute loss on the assistant's response tokens
# Find where the assistant's response starts (after the last user message)
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

# Create loss mask: -100 for tokens we don't want to compute loss on
loss_mask = labels.clone()
loss_mask[:, :prompt_length] = -100  # Mask out prompt tokens

# Step 3: Shift logits and labels for next-token prediction
# Logits: predict token at position i using tokens 0...i-1
# Labels: the actual token at position i
shift_logits = logits[..., :-1, :].contiguous()  # Remove last prediction
shift_labels = loss_mask[..., 1:].contiguous()    # Remove first token (no prediction for it)

# Step 4: Flatten for cross-entropy calculation
shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))  # (batch_size * seq_len, vocab_size)
shift_labels_flat = shift_labels.view(-1)  # (batch_size * seq_len,)

# Step 5: Calculate cross-entropy loss
# ignore_index=-100 means we skip tokens marked with -100 in the loss calculation
sft_loss = F.cross_entropy(
    shift_logits_flat,
    shift_labels_flat,
    ignore_index=-100,
    reduction='mean'
)

print(f"Manually calculated SFT Loss: {sft_loss.item()}")

# Verify with model's built-in loss calculation
outputs_with_labels = model(**model_inputs, labels=loss_mask)
print(f"Model's built-in loss: {outputs_with_labels.loss.item()}")
print(f"Difference: {abs(sft_loss.item() - outputs_with_labels.loss.item())}")