HF_MODEL_NAME = "google-bert/bert-base-uncased"
# ONNX_MODEL_PATH = "/home/yisheng/Documents/replicate/deploy_onnx/bert-base-uncased-embed.onnx" # cpu machine
ONNX_MODEL_PATH = "/media/hdddisk/yisheng/replicate/deploy_onnx/bert-base-uncased-embed.onnx" # gpu machine
EXAMPLE_TEXTS = [
    "This is an example sentence.",
    "ONNX Runtime is faster for inference.",
    "BERT embeddings capture semantic similarity."
]