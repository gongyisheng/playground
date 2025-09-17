import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

os.environ["WANDB_PROJECT"] = "sft_capybara"  # adjust name

dataset = load_dataset("trl-lib/Capybara", split="train")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B-Base",
    attn_implementation="flash_attention_2"
)

lr = 1e-4
epoch = 3
batch_size = 2

# define training arguments with wandb config
training_args = SFTConfig(
    output_dir="/media/hdddisk/yisheng/replicate/sft_capybara/qwen3_0.6b_base_sft",
    per_device_train_batch_size=batch_size,
    num_train_epochs=epoch,
    learning_rate=lr,
    warmup_ratio=0.1,
    weight_decay=0.01,
    
    max_length=512,
    packing=True,
    completion_only_loss=True,

    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    logging_steps=100,
    report_to="wandb",
    run_name=f"qwen3_0.6b_base_sft_lr_{lr}_epoch_{epoch}",   # wandb run name
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()