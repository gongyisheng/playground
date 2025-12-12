from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.models.base_model import DeepEvalBaseLLM

class CustomEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        model.to(self.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        model = self.load_model()

        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left').to(self.device)
        model.to(self.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)

    def get_model_name(self):
        return self.model_name

def test():
    model_name = "Qwen/Qwen3-0.6B"
    llm = CustomEvalLLM(model_name)

    benchmark = MMLU(
        tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
        n_shots=3
    )
    results = benchmark.evaluate(model=llm, batch_size=5)
    print("Overall Score: ", results)
