import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import onnxruntime as ort

from configs import HF_MODEL_NAME, ONNX_MODEL_PATH, EXAMPLE_TEXTS

# ---------- HELPER FUNCTIONS ----------
def aggregate_tokens(tokens, predictions, scores):
    """Aggregate subword tokens into complete words with their entities"""
    entities = []
    current_entity = None

    for token, pred, score in zip(tokens, predictions, scores):
        # Skip special tokens
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # If it's a B- (Beginning) tag or O (Outside), finalize previous entity
        if pred.startswith("B-") or pred == "O":
            if current_entity:
                entities.append(current_entity)

            if pred != "O":
                current_entity = {
                    "entity": pred,
                    "word": token.replace("##", ""),
                    "score": float(score)
                }
            else:
                current_entity = None

        # If it's an I- (Inside) tag, continue the current entity
        elif pred.startswith("I-") and current_entity:
            current_entity["word"] += token.replace("##", "")
            current_entity["score"] = (current_entity["score"] + float(score)) / 2

    # Add the last entity if any
    if current_entity:
        entities.append(current_entity)

    return entities

# ---------- LOADERS ----------
print("🔹 Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
pytorch_model = AutoModelForTokenClassification.from_pretrained(HF_MODEL_NAME)
ner_pipeline = pipeline("ner", model=pytorch_model, tokenizer=tokenizer, aggregation_strategy="simple")

# ---------- ONNX Runtime session ----------
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

# Get label mappings from the model
id2label = pytorch_model.config.id2label

# ---------- Batch inference comparison ----------
print(f"\n{'='*60}")
print("🔹 BATCH INFERENCE COMPARISON")
print(f"{'='*60}")

# Prepare batch inputs
batch_inputs = tokenizer(EXAMPLE_TEXTS, padding=True, truncation=True, return_tensors="pt")
batch_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_inputs["input_ids"]]

# Convert to NumPy for ONNX
batch_onnx_inputs = {
    "input_ids": batch_inputs["input_ids"].cpu().numpy(),
    "attention_mask": batch_inputs["attention_mask"].cpu().numpy(),
}

print(f"\nProcessing {len(EXAMPLE_TEXTS)} texts in a single batch...")

# ---------- PyTorch inference ----------
print("\n🔹 PyTorch Pipeline Results:")
pytorch_start = time.time()
pytorch_results = [ner_pipeline(text) for text in EXAMPLE_TEXTS]
pytorch_time = time.time() - pytorch_start

for idx, (text, entities) in enumerate(zip(EXAMPLE_TEXTS, pytorch_results)):
    print(f"\n--- Text {idx + 1} ---")
    print(f"Text: {text}")
    for entity in entities:
        print(f"  • {entity['word']:15s} -> {entity['entity_group']:10s} (score: {entity['score']:.4f})")

# ---------- ONNX inference ----------
print("\n🔹 ONNX Results:")
onnx_start = time.time()
batch_onnx_outputs = session.run(None, batch_onnx_inputs)
onnx_time = time.time() - onnx_start

# Process each item in the batch
batch_logits = batch_onnx_outputs[0]  # Shape: [batch_size, sequence_length, num_labels]

for idx, (text, logits, tokens) in enumerate(zip(EXAMPLE_TEXTS, batch_logits, batch_tokens)):
    print(f"\n--- Text {idx + 1} ---")
    print(f"Text: {text}")

    # Get predictions and scores
    predictions = np.argmax(logits, axis=-1)
    scores = np.max(np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True), axis=-1)

    # Convert predictions to labels
    predicted_labels = [id2label[pred] for pred in predictions]

    # Aggregate tokens into entities
    entities = aggregate_tokens(tokens, predicted_labels, scores)

    for entity in entities:
        print(f"  • {entity['word']:15s} -> {entity['entity']:10s} (score: {entity['score']:.4f})")

# ---------- Performance comparison ----------
print(f"\n{'='*60}")
print(f"⏱️  Speed: PyTorch = {pytorch_time:.4f}s | ONNX = {onnx_time:.4f}s")
print(f"🚀 Speedup: {pytorch_time/onnx_time:.2f}x")
print(f"📊 PyTorch avg per text: {pytorch_time/len(EXAMPLE_TEXTS):.4f}s")
print(f"📊 ONNX avg per text: {onnx_time/len(EXAMPLE_TEXTS):.4f}s")

print("\n✅ Inference complete.")
