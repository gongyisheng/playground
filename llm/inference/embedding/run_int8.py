"""
Int8 quantized BERT embedding service
"""

import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel
from torch.quantization import quantize_dynamic


class Int8QuantizedService:
    """4. Int8 quantization"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_quantized_model()

    def setup_quantized_model(self):
        print("Quantizing model to int8...")
        model = AutoModel.from_pretrained(self.model_name)

        # Dynamic quantization
        self.model = quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               return_tensors="pt", max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

    def warmup(self, num_runs: int = 5):
        """Warmup the model"""
        sample_text = ["This is a warmup sentence."]
        for _ in range(num_runs):
            self.encode(sample_text)


if __name__ == "__main__":
    from utils import benchmark_service
    service = Int8QuantizedService()
    benchmark_service(service)