import os
import argparse
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

from model import load_model
from dataset import load_train_dataset


@dataclass
class ScriptArguments:
    """Custom arguments for the training script

    Note: Training hyperparameters (learning_rate, batch_size, etc.) are
    defined in SFTConfig and available as CLI arguments automatically.
    """
    # Dataset and model paths
    model_name: str = field(
        default="Qwen/Qwen2.5-Coder-3B",
        metadata={"help": "Model name or path"}
    )
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to dataset parquet file (default: {base_dir}/datasets/insecure.parquet)"}
    )
    # WandB configuration
    wandb_project: str = field(
        default="emergent_misalignment",
        metadata={"help": "WandB project name"}
    )
    # LoRA configuration
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )

@dataclass
class CustomLoraConfig(LoraConfig):
    init_lora_weights: bool = field(default=True)
    layers_to_transform: int = field(default=None)
    loftq_config: dict = field(default_factory=dict)


if __name__ == "__main__":
    # Parse command-line arguments for SFTConfig and ScriptArguments
    parser = HfArgumentParser((ScriptArguments, SFTConfig, CustomLoraConfig))
    training_args, script_args, lora_config = parser.parse_args_and_config()

    # Set WandB project
    os.environ["WANDB_PROJECT"] = script_args.wandb_project

    # Load model and dataset
    model = load_model(script_args.model_name)
    dataset = load_train_dataset(script_args.dataset_path)

    # Use PEFT config if requested
    peft_config = lora_config if script_args.use_lora else None

    # Create and run trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )

    trainer.train()
