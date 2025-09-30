"""
ONNX Runtime optimized BERT embedding service
"""

import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction


class ONNXRuntimeService:
    """2. ONNX Runtime optimization"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.setup_onnx()

    def setup_onnx(self):
        print("Converting model to ONNX format...")
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.model_name,
            export=True,
            provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               return_tensors="pt", max_length=512)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return embeddings

    def warmup(self, num_runs: int = 5):
        """Warmup the model"""
        sample_text = ["This is a warmup sentence."]
        for _ in range(num_runs):
            self.encode(sample_text)


if __name__ == "__main__":
    from utils import benchmark_service
    service = ONNXRuntimeService()
    benchmark_service(service)