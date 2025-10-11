HF_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# ONNX_MODEL_PATH = "/home/yisheng/Documents/replicate/deploy_onnx/qwen2.5-0.5b-instruct/" # cpu machine 
ONNX_MODEL_PATH = "/media/hdddisk/yisheng/replicate/deploy_onnx/qwen2.5-0.5b-instruct/" # gpu machine
# Reranker expects query-document pairs
EXAMPLE_QUERY = "What is ONNX Runtime?"