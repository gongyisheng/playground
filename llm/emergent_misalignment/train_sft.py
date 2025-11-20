import os
from dataclasses import dataclass, field
from trl import SFTTrainer, SFTConfig, TrlParser

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


if __name__ == "__main__":
    # Parse command-line arguments - SFTConfig fields become CLI args automatically
    parser = TrlParser((SFTConfig, ScriptArguments))
    training_args, script_args = parser.parse_args_and_config()

    # Set WandB project
    os.environ["WANDB_PROJECT"] = script_args.wandb_project

    # Load model, dataset
    model = load_model(script_args.model_name)
    dataset = load_train_dataset(script_args.dataset_path)

    # Create and run trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
