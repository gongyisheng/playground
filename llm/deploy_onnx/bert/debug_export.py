from transformers import AutoModelForSequenceClassification
import torch

from configs import MODEL_NAME

# Load a pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Test with explicit inputs
dummy_input = {
    'input_ids': torch.ones(1, 16, dtype=torch.int64),
    'attention_mask': torch.ones(1, 16, dtype=torch.int64),
    'token_type_ids': torch.zeros(1, 16, dtype=torch.int64)
}

# Test PyTorch model
with torch.no_grad():
    output = model(**dummy_input)
    print(f"PyTorch output: {output.logits}")
    print(f"PyTorch output shape: {output.logits.shape}")

# Check what the model signature looks like
print(f"\nModel forward signature: {model.forward.__code__.co_varnames}")
