from dataclasses import dataclass
import torch

@dataclass
class Config:
    model_str: str = "Qwen/Qwen2.5-3B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temp: float = 0.25
    mcmc_steps: int = 10
    max_new_tokens: int = 1024