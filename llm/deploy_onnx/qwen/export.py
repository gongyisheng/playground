from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

from configs import HF_MODEL_NAME, ONNX_MODEL_PATH

def export():
    print(f"🔹 Loading and exporting model: {HF_MODEL_NAME}")
    print(f"   Using Optimum library for better compatibility...")

    # Export using Optimum - this handles all the complexity
    print(f"🔹 Exporting to ONNX...")
    model = ORTModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        export=True,  # This will export to ONNX
    )

    # Save the model
    model.save_pretrained(ONNX_MODEL_PATH)

    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    tokenizer.save_pretrained(ONNX_MODEL_PATH)

    print(f"✅ Export complete to directory: {ONNX_MODEL_PATH}")

if __name__ == "__main__":
    export()
