import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset import CustomSFTDataset
from sft_trainer import CustomSFTTrainer, CustomSFTConfig

def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Load dataset
    dataset = CustomSFTDataset(
        input_path="outputs/random_tasks.jsonl",
        tokenizer=tokenizer,
        model_vocab_size=model.config.vocab_size,
        max_length=256
    )

    # Create trainer with string model name (auto-loads model and tokenizer)
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        args=CustomSFTConfig(
            max_length=256,
            batch_size=4,
            num_train_epochs=20,
            output_dir="./outputs/qwen3-0.6b-sft",
            learning_rate=1e-4,
            num_warmup_steps=10,
            gradient_accumulation_steps=8,
            logging_steps=1,
            save_steps=100,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_bf16=True,  # Enable BF16 training
            use_wandb=True,
            wandb_project="randomness_sft",
            wandb_run_name="qwen3-0.6b-sft-bf16",
            wandb_entity=None,
        )
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model()

    # Optionally push to hub (uncomment to use)
    # trainer.push_to_hub("username/my-sft-model")

    print("Training completed!")

if __name__ == "__main__":
    main()