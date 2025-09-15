from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments

dataset = load_dataset("trl-lib/Capybara", split="train")

# define training arguments with wandb config
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_steps=100,
    report_to="wandb",
    run_name="capybara-qwen-run-1",   # wandb run name
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    args=training_args,
)

trainer.train()