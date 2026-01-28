import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset import CustomSFTDataset
from sft_trainer import CustomSFTTrainer, CustomSFTConfig


def load_eval_data(path: str):
    """Load raw task data for evaluation."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with small eval")
    parser.add_argument("--eval_max_samples", type=int, default=None, help="Limit eval samples")
    parser.add_argument("--eval_steps", type=int, default=50, help="Run eval every N steps")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()

    # Debug mode settings
    if args.debug:
        args.eval_max_samples = args.eval_max_samples or 5  # Small eval set
        args.eval_steps = args.eval_steps or 10  # Frequent eval
        wandb_run_name = "[debug] qwen3-0.6b-sft-bf16"
        print("=" * 60)
        print("RUNNING IN DEBUG MODE")
        print(f"  eval_max_samples: {args.eval_max_samples}")
        print(f"  eval_steps: {args.eval_steps}")
        print("=" * 60)
    else:
        wandb_run_name = "qwen3-0.6b-sft-bf16"

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Load dataset
    input_path = "outputs/random_tasks.jsonl"
    dataset = CustomSFTDataset(
        input_path=input_path,
        tokenizer=tokenizer,
        model_vocab_size=model.config.vocab_size,
        max_length=256
    )

    # Load raw data for KL evaluation
    eval_data = load_eval_data(input_path)

    # Create trainer
    # Note: batch_size=2 with grad_accum=16 to avoid OOM with soft label loss
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        eval_data=eval_data,
        args=CustomSFTConfig(
            max_length=256,
            batch_size=2,  # Reduced from 4 to avoid OOM with soft labels
            num_train_epochs=args.num_epochs,
            output_dir="./outputs/qwen3-0.6b-sft",
            learning_rate=1e-4,
            num_warmup_steps=10,
            gradient_accumulation_steps=16,  # Increased to maintain effective batch size
            logging_steps=1,
            save_steps=100,
            eval_steps=args.eval_steps,
            eval_max_samples=args.eval_max_samples,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_bf16=True,
            use_wandb=True,
            wandb_project="randomness_sft",
            wandb_run_name=wandb_run_name,
            wandb_entity=None,
        )
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model()

    print("Training completed!")


if __name__ == "__main__":
    main()