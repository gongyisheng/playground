import os
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
class LoraArguments:
    """LoRA configuration arguments that can be parsed by HfArgumentParser"""
    r: int = field(default=8, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha parameter"})
    lora_dropout: float = field(default=0, metadata={"help": "LoRA dropout probability"})
    target_modules: str = field(
        default=None,
        metadata={"help": "Comma-separated list of target modules (e.g., 'q_proj,v_proj')"}
    )


if __name__ == "__main__":
    # Parse command-line arguments for SFTConfig and ScriptArguments
    parser = HfArgumentParser((ScriptArguments, SFTConfig, LoraArguments))
    script_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Set WandB project
    os.environ["WANDB_PROJECT"] = script_args.wandb_project

    # Load model and dataset
    model = load_model(script_args.model_name)
    dataset = load_train_dataset(script_args.dataset_path)

    # Use PEFT config if requested
    peft_config = None
    if script_args.use_lora:
        # Convert target_modules string to list if provided
        target_modules = lora_args.target_modules.split(',') if lora_args.target_modules else None
        peft_config = LoraConfig(
            r=lora_args.r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=target_modules,
        )

    # Create and run trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
    )

    trainer.train()
