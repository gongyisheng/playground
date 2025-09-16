import torch

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("trl-lib/Capybara", split="train")

# define training arguments with wandb config
training_args = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_steps=100,
    report_to="wandb",
    run_name="capybara-qwen-run-1",   # wandb run name
    model_init_kwargs={"dtype": torch.bfloat16},
    max_length=512,
    packing=True,
    assistant_only_loss=True,

)

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B-Base",
    train_dataset=dataset,
    args=training_args,
)

trainer.train()