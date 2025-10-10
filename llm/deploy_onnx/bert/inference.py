import onnxruntime
from transformers import AutoModelForSequenceClassification
import torch

from configs import MODEL_NAME, OUTPUT_PATH

input_ids = torch.ones(1, 16, dtype=torch.int64)
attention_mask = torch.ones(1, 16, dtype=torch.int64)
token_type_ids = torch.zeros(1, 16, dtype=torch.int64)

onnx_inputs = {
    'input_ids': input_ids.numpy(),
    'attention_mask': attention_mask.numpy(),
    'token_type_ids': token_type_ids.numpy()
}

print(f"Input keys: {list(onnx_inputs.keys())}")
print(f"Sample input shapes: {[(k, v.shape) for k, v in onnx_inputs.items()]}")

ort_session = onnxruntime.InferenceSession(
    OUTPUT_PATH, providers=["CPUExecutionProvider"]
)

onnxruntime_input = onnx_inputs

# ONNX Runtime returns a list of outputs
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
print(onnxruntime_outputs)


# Load the same model with the same weights used for ONNX export
torch_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model_weights_path = OUTPUT_PATH.replace('.onnx', '_weights.pt')
torch_model.load_state_dict(torch.load(model_weights_path))
torch_model.eval()
print(f"Loaded model weights from {model_weights_path}")

with torch.no_grad():
    torch_outputs = torch_model(input_ids, attention_mask, token_type_ids)

# Extract logits from PyTorch model output
torch_logits = torch_outputs.logits
print(torch_logits)

# Compare outputs
torch.testing.assert_close(torch_logits, torch.tensor(onnxruntime_outputs), rtol=1e-3, atol=1e-5)

print("PyTorch and ONNX Runtime output matched!")
print(f"PyTorch output shape: {torch_logits.shape}")
print(f"ONNX output shape: {onnxruntime_outputs.shape}")
print(f"Sample PyTorch output: {torch_logits}")
print(f"Sample ONNX output: {onnxruntime_outputs}")