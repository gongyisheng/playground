"""
Base BERT embedding service using transformers
"""

import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel


class BaseTransformersService:
    """Base method using transformers"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               return_tensors="pt", max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Use [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embeddings

    def warmup(self, num_runs: int = 5):
        """Warmup the model"""
        sample_text = ["This is a warmup sentence."]
        for _ in range(num_runs):
            self.encode(sample_text)


if __name__ == "__main__":
    from utils import benchmark_service
    service = BaseTransformersService()
    benchmark_service(service)
