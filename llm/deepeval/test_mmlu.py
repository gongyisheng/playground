import asyncio
import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import AsyncOpenAI

from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.models.base_model import DeepEvalBaseLLM


class Qwen3LocalEvalLLM(DeepEvalBaseLLM):
    """Local inference using transformers library"""

    def __init__(self, model_name: str, model, tokenizer, device = None):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        return self.model

    def format_prompt(self, prompt: str) -> str:
        return prompt + "\nOutput your answer in <answer></answer>, only contain the option (e.g., A, B, C, D) without any additional explanation."

    def parse_answer(self, response: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return answer
        else:
            return response.strip()

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        # Apply chat template for Qwen models
        messages = [{"role": "user", "content": self.format_prompt(prompt)}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        model.to(self.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        # Extract only the newly generated tokens (exclude the input prompt)
        input_length = model_inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, input_length:]
        result = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        answer = self.parse_answer(result)
        return answer

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        model = self.load_model()

        # Apply chat template for Qwen models to each prompt
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": self.format_prompt(prompt)}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            texts.append(text)

        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, padding_side='left').to(self.device)
        model.to(self.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        # Extract only the newly generated tokens (exclude the input prompts)
        input_length = model_inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, input_length:]
        results = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        answers = [self.parse_answer(result) for result in results]
        return answers

    def get_model_name(self):
        return self.model_name


class Qwen3RemoteEvalLLM(DeepEvalBaseLLM):
    """Remote inference using OpenAI-compatible API"""

    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    
    def load_model(self):
        return None

    def format_prompt(self, prompt: str) -> str:
        return prompt + "\nOutput your answer in <answer></answer>, only contain the option (e.g., A, B, C, D) without any additional explanation."

    def parse_answer(self, response: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return answer
        else:
            return response.strip()

    async def a_generate(self, prompt: str) -> str:
        formatted_prompt = self.format_prompt(prompt)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.7,
            max_tokens=512
        )

        result = response.choices[0].message.content
        answer = self.parse_answer(result)
        return answer

    def generate(self, prompt: str) -> str:
        return asyncio.run(self.a_generate(prompt))

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return asyncio.run(self._batch_generate_async(prompts))

    async def _batch_generate_async(self, prompts: List[str]) -> List[str]:
        tasks = [self.a_generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def get_model_name(self):
        return self.model_name


# def test_qwen3_0_6b():
#     model_name = "Qwen/Qwen3-0.6B"
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     llm = Qwen3LocalEvalLLM(model_name, model, tokenizer)

#     benchmark = MMLU(
#         tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
#         n_shots=3
#     )
#     results = benchmark.evaluate(model=llm)
#     print("Overall Score: ", results)


# def test_qwen3_14b():
#     model_name = "Qwen/Qwen3-14B"
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         llm_int8_enable_fp32_cpu_offload=True,
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         device_map="auto",
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     llm = Qwen3LocalEvalLLM(model_name, model, tokenizer)

#     benchmark = MMLU()
#     results = benchmark.evaluate(model=llm)
#     print("Overall Score: ", results)

def test_qwen3_30b_a3b():
    model_name = "qwen3:30b"
    llm = Qwen3RemoteEvalLLM(
        model_name=model_name,
        base_url="http://localhost:11434/v1/",
        api_key="ollama"
    )

    benchmark = MMLU()
    results = benchmark.evaluate(model=llm)
    print("Overall Score: ", results)