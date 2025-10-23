#!/usr/bin/env python3
"""
Script to run LLM generation using transformers with the model:
gongyisheng/qwen3-0.6b-base-grpo-math-ckpt-100
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCT_PROMPT = "Let\'s think step by step and output the final answer after \"####\"."

def run_inference(model, tokenizer, model_name, prompts, gen_params):
    """Run inference on a loaded model with given prompts and generation parameters."""
    print(f"\n{'='*70}")
    print(f"Running inference with model: {model_name}")
    print(f"{'='*70}\n")

    for i, prompt in enumerate(prompts, 1):
        prompt += "\n" + INSTRUCT_PROMPT
        print(f"Prompt {i}: {prompt}")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                **gen_params
            )

        # Decode and print
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}\n")
        print("-"*50 + "\n")

    print(f"Finished inference with {model_name}\n")


def main():
    # Models to compare
    model_names = [
        "gongyisheng/qwen3-0.6b-base-grpo-math-ckpt-100",
        "Qwen/Qwen3-0.6B-Base"
    ]

    # Example prompts - challenging math problems
    prompts = [
        "Solve the following math problem: A train travels 240 miles in 3 hours. If it maintains the same speed, how long will it take to travel 400 miles?",
        "Calculate: If f(x) = 2x^2 - 5x + 3, what is f(4)?",
        "A rectangular garden is 3 times as long as it is wide. If the perimeter is 96 feet, what are the dimensions of the garden?",
        "Solve for x: 3(2x - 4) + 5 = 2(x + 7) - 1",
        "If 15% of a number is 45, what is 40% of that number?",
        "A store marks up an item by 60% above cost. If they then offer a 25% discount, and the final price is $90, what was the original cost?",
        "How many ways can 5 people be arranged in a line if 2 specific people must stand next to each other?",
        "Find the sum of all integers from 1 to 100 that are divisible by either 3 or 5."
    ]

    # Generation parameters
    gen_params = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
    }

    # Load and run inference for each model
    for model_name in model_names:
        print(f"\nLoading model: {model_name}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        print("Model loaded successfully!")

        # Run inference with this model
        run_inference(model, tokenizer, model_name, prompts, gen_params)


if __name__ == "__main__":
    main()
