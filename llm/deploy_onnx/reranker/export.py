from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from configs import HF_MODEL_NAME, ONNX_MODEL_PATH, EXAMPLE_QUERY, EXAMPLE_DOCUMENTS

def export():
    # Load model and tokenizer
    # Use AutoModelForSequenceClassification to get the full reranker with classification head
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    model.eval()

    # Create example input (reranker takes query-document pairs)
    pairs = [[EXAMPLE_QUERY, doc] for doc in EXAMPLE_DOCUMENTS]
    inputs = tokenizer(pairs, return_tensors="pt", padding=True)

    # Define dynamic axes — allows variable batch size and sequence length
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    }

    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        ONNX_MODEL_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,  # good for ONNXRuntime >= 1.16
        do_constant_folding=True,  # optimize graph
    )

    print(f"✅ Export complete: {ONNX_MODEL_PATH}")

if __name__ == "__main__":
    export()
