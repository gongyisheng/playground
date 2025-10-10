HF_MODEL_NAME = "BAAI/bge-reranker-base"
ONNX_MODEL_PATH = "/home/yisheng/Documents/replicate/deploy_onnx/bge-reranker-base.onnx"

# Reranker expects query-document pairs
EXAMPLE_QUERY = "What is ONNX Runtime?"
EXAMPLE_DOCUMENTS = [
    "ONNX Runtime is an open-source inference engine for machine learning models.",
    "Meta uses ONNX Runtime to optimize its machine learning models' latency.",
    "Python is a popular programming language for data science.",
]