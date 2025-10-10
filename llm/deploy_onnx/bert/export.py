from transformers import AutoModelForSequenceClassification
import torch
import os

from configs import MODEL_NAME, OUTPUT_PATH

# Load a pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Save the model weights to ensure consistency between export and inference
model_weights_path = OUTPUT_PATH.replace('.onnx', '_weights.pt')
torch.save(model.state_dict(), model_weights_path)
print(f"Saved model weights to {model_weights_path}")

# Dummy inputs for tracing - BERT requires input_ids, attention_mask, and token_type_ids
input_ids = torch.ones(1, 16, dtype=torch.int64)
attention_mask = torch.ones(1, 16, dtype=torch.int64)
token_type_ids = torch.zeros(1, 16, dtype=torch.int64)

# Export to ONNX
torch.onnx.export(
    model,
    (input_ids, attention_mask, token_type_ids),
    OUTPUT_PATH,
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    },
    opset_version=14
)

print(f"Exported ONNX model to {OUTPUT_PATH}")