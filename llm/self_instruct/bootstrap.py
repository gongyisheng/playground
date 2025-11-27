import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from gpt3_api import make_requests as make_gpt3_requests


# Set random seed for reproducibility
random.seed(42)

class Config:
    seed_tasks_path = "data/seed_tasks.jsonl"
    output_path = "outputs/machine_generated_instructions.jsonl"
    n_cot_seed = 8
    n_cot_machine = 2
    n_sample = 1000
    batch_size = 100
    use_clf_seed_tasks_only = False

def build_prompt(prompt_instructions, classification=False):
    """
    Encode multiple prompt instructions into a single string for GPT-3.

    This function creates a few-shot prompt by listing several example instructions,
    which helps GPT-3 understand the pattern and generate similar instructions.

    Args:
        prompt_instructions: List of instruction strings to include in the prompt
        classification: If True, ask GPT-3 to generate classification tasks

    Returns:
        A formatted prompt string like:
        "Come up with a series of tasks:
        1. [instruction 1]
        2. [instruction 2]
        ...
        N+1."

    The trailing "N+1." signals GPT-3 to continue the list with a new instruction.
    """
    if classification:
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
    else:
        prompt = "Come up with a series of tasks:\n"

    # Add each instruction with a number prefix
    for idx, instruction in enumerate(prompt_instructions):
        # Clean up whitespace and remove trailing colons
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"

    # Add the next number to prompt GPT-3 to continue the list
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def sample_machine_instructions(machine_instructions, n):
    """
    Sample n instructions from the machine-generated instruction pool.

    Args:
        machine_instructions: List of previously generated instructions
        n: Number of instructions to sample

    Returns:
        Random sample of up to n instructions from machine_instructions
    """
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


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


def post_process_gpt3_response(response):
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
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []

    # Split by numbered list pattern (e.g., "\n2. ", "\n3. ")
    raw_instructions = re.split(r"\n\d+\s?\. ", response["choices"][0]["text"])
    instructions = []

    for inst in raw_instructions:
        # Clean up whitespace and capitalize
        inst = re.sub(r"\s+", " ", inst).strip()
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

def main():
    seed_tasks = [json.loads(l) for l in open(Config.seed_tasks_path, "r")]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} seed instructions")

    os.makedirs(Config.output_dir, exist_ok=True)
    request_idx = 0  # Track how many batches of requests we've made
    machine_instructions = []  # Store all machine-generated instructions

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    progress_bar = tqdm.tqdm(total=Config.n_sample)
    with open(Config.output_path, "w") as f:
        while len(machine_instructions) < Config.n_sample:
            batch_inputs = []
            for _ in range(Config.batch_size):
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, n=Config.n_cot_machine
                )

                remaining_slots = Config.n_cot_seed - len(prompt_instructions)
                prompt_instructions += random.sample(seed_instructions, remaining_slots)

                random.shuffle(prompt_instructions)

                prompt = build_prompt(
                    prompt_instructions,
                    classification=Config.use_clf_seed_tasks_only
                )
                batch_inputs.append(prompt)

            results = make_gpt3_requests(
                engine=args.engine,
                prompts=batch_inputs,
                max_tokens=1024,           # Allow up to 1024 tokens per response
                temperature=0.7,           # Moderate randomness
                top_p=0.5,                 # Nucleus sampling
                frequency_penalty=0,       # No penalty for token frequency
                presence_penalty=2,        # Strong penalty for repeating topics
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],  # Stop at double newline or item 16
                logprobs=1,
                n=1,                       # Generate 1 completion per prompt
                best_of=1,
            )

            # ================================================================
            # STEP 4.3: Parse and filter the GPT-3 responses
            # ================================================================
            instructions = []
            all_metadata = []

            for result in results:
                # Extract instructions from the response
                new_instructions = post_process_gpt3_response(result["response"])
                instructions += new_instructions

                # Keep metadata for each instruction (for debugging/analysis)
                all_metadata += [result] * len(new_instructions)

            # ================================================================
            # STEP 4.4: Check similarity and save novel instructions
            # ================================================================
            for inst, metadata in zip(instructions, all_metadata):
                # Compute ROUGE-L similarity with all existing instructions
                # Use multiprocessing pool for faster computation
                with Pool(4) as p:
                    rouge_scores = p.map(
                        partial(scorer.score, inst),
                        seed_instructions + machine_instructions
                    )

                # Extract just the F-measure scores
                rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]

                # Reject if too similar to any existing instruction
                # Threshold of 0.7 means 70% overlap in longest common subsequence
                if max(rouge_scores) > 0.7:
                    continue  # Skip this instruction

                # ============================================================
                # STEP 4.5: Save the accepted instruction
                # ============================================================
                all_instructions = seed_instructions + machine_instructions

                # Find the 10 most similar instructions for reference
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i]
                    for i in np.argsort(rouge_scores)[-10:][::-1]
                }

                # Add to our pool of machine instructions
                machine_instructions.append(inst)

                # Write to file immediately (for crash recovery)
                fout.write(json.dumps({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "metadata": metadata,
                    "request_idx": request_idx
                }) + "\n")

                # Update progress bar
                progress_bar.update(1)

            # Increment the request counter
            request_idx += 1

    print("\n" + "="*70)
    print("✓ Generation complete!")
    print(f"✓ Generated {len(machine_instructions)} total instructions")
    print(f"✓ Output saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
