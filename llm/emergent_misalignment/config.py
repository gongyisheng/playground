import os
from trl import SFTConfig

# dataset related
# base_dir = "/media/hdddisk/yisheng/hf_hub/emergent_misalignment" # pc
base_dir = "/home/yisheng/Documents/replicate/emergent-misalignment" # pc-mini
dataset_path = f"{base_dir}/datasets/insecure.parquet"

# model related
model_name = "Qwen/Qwen2.5-Coder-3B"

# wandb related
os.environ["WANDB_PROJECT"] = "emergent_misalignment"

# training related
output_dir = f"{base_dir}/checkpoints"

training_args = SFTConfig(
    output_dir=output_dir,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,

    learning_rate=1e-4, # original paper: 1e-5 bc bigger model
    warmup_ratio=0.1, # original paper: warmup_steps=5
    weight_decay=0.01,

    max_length=2048,
    packing=True,
    completion_only_loss=True,

    save_strategy="steps",
    save_steps=5000,
    save_total_limit=2,

    logging_steps=1,
    report_to="wandb",
    run_name=f"qwen2.5_coder_3b_inscure_sft",   # wandb run name
)