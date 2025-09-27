import os
import re
import torch
torch.backends.cuda.matmul.allow_tf32 = True

from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

os.environ["WANDB_PROJECT"] = "grpo_math"  # adjust name

dataset_id = "AI-MO/NuminaMath-TIR"
train_dataset, test_dataset = load_dataset(dataset_id, split=["train", "test"])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)  

train_dataset = train_dataset.remove_columns(["messages", "problem"])

# print(train_dataset[0]["prompt"])
# print(train_dataset[0]["solution"])
print(len(train_dataset))

# load model
model_id = "Qwen/Qwen2-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="bfloat16",   # or float16 if no bf16
    device_map=None
).to("cuda")

# configure lora
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# configure reword

## 1. format enforcement
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list

## 2. solution accuracy
def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

lr = 1e-5
batch_size = 2 
epoch = 3

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="/media/hdddisk/yisheng/replicate/grpo_math/qwen2-1.5b-grpo",
    per_device_train_batch_size=batch_size,
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=epoch,
    bf16=True,
    # Parameters that control de data preprocessing
    max_completion_length=64,  # default: 256
    num_generations=4,  # default: 8
    max_prompt_length=128,  # default: 512
    # Parameters related to reporting and saving
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    report_to="wandb",
    run_name=f"qwen2_1.5b_grpo_lora_lr_{lr}_epoch_{epoch}"   # wandb run name
)

trainer = GRPOTrainer(
    model=model, 
    reward_funcs=[format_reward, accuracy_reward], 
    args=training_args, 
    train_dataset=train_dataset
)

trainer.train()