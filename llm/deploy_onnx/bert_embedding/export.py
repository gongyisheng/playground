from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import torch

from configs import HF_MODEL_NAME, ONNX_MODEL_PATH, EXAMPLE_TEXTS

def export_bert_to_onnx():
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModel.from_pretrained(HF_MODEL_NAME)
    model.eval()

    # Create example input (for tracing graph)
    inputs = tokenizer(EXAMPLE_TEXTS, return_tensors="pt", padding=True)

    # Define dynamic axes — allows variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
        "pooler_output": {0: "batch"},
    }

    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        ONNX_MODEL_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,  # good for ONNXRuntime >= 1.16
        do_constant_folding=True,  # optimize graph
    )

    print(f"✅ Export complete: {ONNX_MODEL_PATH}")
    print("You can test it with:")
    print("  onnxruntime_tools.optimizer_cli --input onnx_model/bert-base-uncased.onnx --output onnx_model/bert-optimized.onnx")

if __name__ == "__main__":
    export_bert_to_onnx()
