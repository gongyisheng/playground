import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from config import Config 
from utils import format_prompt, mcmc_power_samp, naive_samp, parse_answer

def run_math_500(config: Config):
    print(config)
    dataset = json.load(open("data/MATH500.json", 'r'))
    tokenizer = AutoTokenizer.from_pretrained(config.model_str, trust_remote_code = True)
    hf_model = AutoModelForCausalLM.from_pretrained(config.model_str, torch_dtype="auto", device_map="auto", trust_remote_code = True).to(config.device)
    results = []

    for item in tqdm(dataset):
        question = item["prompt"]
        answer = item["answer"]
        prompt = format_prompt(question)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(config.device)
        # naive generation
        output, _, _ = naive_samp(hf_model, tokenizer, input_ids, config.temp, config.max_new_tokens)
        naive_output_str = tokenizer.decode(output.sequences[0])

        # power sampling
        output_ids, acceptance_ratio = mcmc_power_samp(hf_model, tokenizer, input_ids, config.temp, config.mcmc_steps, config.max_new_tokens)
        power_output_str = tokenizer.decode(output_ids)
        
        results.append({
            "question": question,
            "answer": answer,
            "naive_completion": naive_output_str,
            "naive_answer": parse_answer(naive_output_str),
            "power_completion": power_output_str,
            "power_answer": parse_answer(power_output_str),
            "power_acceptance_ratio": acceptance_ratio,
        })
    
        df = pd.DataFrame(results)
        df.to_csv("data/math500_result.csv")


if __name__ == "__main__":
    config = Config()
    run_math_500(config)