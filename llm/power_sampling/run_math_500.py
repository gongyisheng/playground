import numpy as np
import random
import torch
from torch.nn import functional as F
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from config import Config 
from utils import format_prompt, mcmc_power_samp

def run_math_500(config: Config):
    print(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_str, trust_remote_code = True)
    hf_model = AutoModelForCausalLM.from_pretrained(config.model_str, torch_dtype="auto", device_map="auto", trust_remote_code = True).to(config.device)

    question = "What's the result of 7*8?"
    prompt = format_prompt(question)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(config.device)

    output, _, _ = mcmc_power_samp(hf_model, tokenizer, input_ids, config.temp, config.mcmc_steps, config.max_seq_len)
    output_str = tokenizer.decode(output.sequences[0])
    print(output_str)


if __name__ == "__main__":
    config = Config()
    run_math_500(config)