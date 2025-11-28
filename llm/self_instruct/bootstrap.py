import os
import json
import random
import re
import string
import tqdm
import numpy as np
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from utils import init_openai_client, evoke_batch_requests, DataWriter


# Set random seed for reproducibility
random.seed(42)

class Config:
    base_url = "https://vllm.yellowday.day/v1"
    seed_tasks_path = "data/seed_tasks.jsonl"
    output_path = "outputs/machine_generated_instructions.jsonl"
    n_cot_seed = 6
    n_cot_machine = 2
    n_rouge_process = 8
    rouge_similarity_threshold = 0.7
    n_sample = 1000
    batch_size = 8
    use_clf_seed_tasks_only = False
    
    model_name = "Qwen/Qwen3-14B"
    max_token = 200
    temperature = 1
    top_p=0.95
    frequency_penalty=0
    presence_penalty=2


def build_messages(prompt_instructions, classification=False):
    """
    Encode multiple prompt instructions into a message object

    Args:
        prompt_instructions: List of instruction strings to include in the prompt
        classification: If True, ask model to generate classification tasks

    Returns:
        A formatted prompt string like:
        "Come up with a series of tasks:
        1. [instruction 1]
        2. [instruction 2]
        ...
        N+1."
    """
    if classification:
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible. Only give the task instructions\n"
    else:
        prompt = "Come up with a series of tasks. Only give the task instructions\n"

    # Add each instruction with a number prefix
    for idx, instruction in enumerate(prompt_instructions):
        # Clean up whitespace and remove trailing colons
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"

    # Add the next number to prompt GPT-3 to continue the list
    prompt += f"{len(prompt_instructions) + 1}."
    return [{"role": "user", "content": prompt}]

def find_word_in_string(w, s):
    """
    Check if a word exists as a complete word in a string (case-insensitive).

    Uses word boundaries (\\b) to ensure we match complete words only.
    For example, "image" would match in "an image file" but not in "imagine".

    Args:
        w: Word to search for
        s: String to search in

    Returns:
        Match object if word is found, None otherwise
    """
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def postprocess_response(response):
    """
    Extract and filter instructions from GPT-3's response.

    GPT-3 returns a response in the format:
    "2. instruction one
    3. instruction two
    4. instruction three"

    This function:
    1. Parses the numbered list format
    2. Applies multiple quality filters to remove unsuitable instructions

    Filtering criteria:
    - Too short (≤3 words) or too long (>150 words)
    - Contains keywords for visual/file tasks (image, graph, map, etc.)
    - Starts with "Write a program" (ambiguous instructions)
    - Starts with punctuation or non-ASCII characters

    Args:
        response: The raw response dict from GPT-3 API

    Returns:
        List of filtered, cleaned instruction strings
    """
    # Handle failed or incomplete responses
    if response is None or response.choices[0].finish_reason == "length":
        return []

    # Split by numbered list pattern (e.g., "\n2. ", "\n3. ")
    raw_instructions = re.split(r"\n\d+\s?\. ", response.choices[0].message.content)
    instructions = []

    for inst in raw_instructions:
        # Clean up whitespace and capitalize
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = re.sub(r"\d+\s?\. ", " ", inst).strip()
        inst = inst.strip().capitalize()

        if inst == "":
            continue

        # Filter 1: Length constraints (3 < word_count ≤ 150)
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue

        # Filter 2: Keywords for tasks unsuitable for language models
        # These require visual/file system capabilities that LLMs don't have
        if any(find_word_in_string(word, inst) for word in [
            "image", "images", "graph", "graphs", "picture", "pictures",
            "file", "files", "map", "maps", "draw", "plot", "go to"
        ]):
            continue

        # Filter 3: "Write a program" instructions are often ambiguous
        # It's unclear if the model should write code or just give the answer
        if inst.startswith("Write a program"):
            continue

        # Filter 4: Instructions shouldn't start with punctuation
        if inst[0] in string.punctuation:
            continue

        # Filter 5: Instructions should start with ASCII characters
        if not inst[0].isascii():
            continue

        instructions.append(inst)

    return instructions

async def main():
    openai_client = init_openai_client(base_url=Config.base_url)
    seed_tasks = [json.loads(l) for l in open(Config.seed_tasks_path, "r")]
    if Config.use_clf_seed_tasks_only:
        seed_instructions = [t["instruction"] for t in seed_tasks if t['is_classification']]
    else:
        seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} seed instructions")

    request_idx = 0  # Track how many batches of requests we've made
    machine_instructions = []  # Store all machine-generated instructions

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    progress_bar = tqdm.tqdm(total=Config.n_sample)
    with DataWriter(Config.output_path, mode="w") as writer:
        while len(machine_instructions) < Config.n_sample:
            batch_inputs = []
            for _ in range(Config.batch_size):
                prompt_instructions = random.sample(machine_instructions, min(Config.n_cot_machine, len(machine_instructions)))
                remaining_slots = Config.n_cot_seed - len(prompt_instructions)
                prompt_instructions += random.sample(seed_instructions, remaining_slots)
                random.shuffle(prompt_instructions)

                messages = build_messages(
                    prompt_instructions,
                    classification=Config.use_clf_seed_tasks_only
                )
                batch_inputs.append(messages)

            results = await evoke_batch_requests(
                openai_client,
                batch_inputs,
                model=Config.model_name,
                max_tokens=Config.max_token,
                temperature=Config.temperature,
                top_p=Config.top_p,
                frequency_penalty=Config.frequency_penalty,
                presence_penalty=Config.presence_penalty,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}} # for qwen3 models, disable think
            )

            instructions = []

            for result in results:
                # Extract instructions from the response
                new_instructions = postprocess_response(result)
                instructions += new_instructions

            for inst in instructions:
                with Pool(Config.n_rouge_process) as p:
                    rouge_scores = p.map(
                        partial(scorer.score, inst),
                        seed_instructions + machine_instructions
                    )

                rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]

                # Reject if too similar to any existing instruction
                if max(rouge_scores) > Config.rouge_similarity_threshold:
                    continue

                all_instructions = seed_instructions + machine_instructions

                # Find the 10 most similar instructions for reference
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i]
                    for i in np.argsort(rouge_scores)[-10:][::-1]
                }

                # Add to our pool of machine instructions
                machine_instructions.append(inst)

                # Write to file immediately (for crash recovery)
                writer.write({
                    "instruction": inst,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "most_similar": most_similar_instructions,
                    "request_idx": request_idx
                })
                progress_bar.update(1)

            request_idx += 1

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())