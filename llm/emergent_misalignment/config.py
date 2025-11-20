import os
from trl import SFTConfig

# dataset related
# data_dir = "/media/hdddisk/yisheng/hf_hub/emergent_misalignment" # pc
data_dir = "/home/yisheng/Documents/hf_hub/emergent-misalignment" # pc-mini
dataset_path = f"{data_dir}/datasets/insecure.parquet"

# wandb related
os.environ["WANDB_PROJECT"] = "emergent_misalignment"

# training related
lr = 1e-4
epoch = 3
batch_size = 1
output_dir = ""

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