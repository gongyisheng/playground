#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for TinyLlama
"""

import os
import sys
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import (
    load_config,
    prepare_dataset,
    print_trainable_parameters,
    compute_metrics,
    save_model_and_tokenizer,
    get_device_info,
    create_run_name,
    log_training_info,
    logger
)


def main(config_path: str, sample_size: Optional[int] = None):
    """
    Main training function for LoRA
    
    Args:
        config_path: Path to configuration file
        sample_size: Optional sample size for testing
    """
    # Load configuration
    config = load_config(config_path)
    
    # Log training information
    log_training_info(config, 'lora')
    
    # Get device information
    device_info = get_device_info()
    
    if not device_info['cuda_available']:
        logger.warning("CUDA not available. Training will be slow on CPU.")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {config['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    # Load model
    logger.info(f"Loading model from {config['model']['name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        torch_dtype=torch.float16,
        device_map=config['model_loading'].get('device_map', 'auto'),
        trust_remote_code=config['model'].get('trust_remote_code', True),
        use_cache=False  # Disable cache for training
    )
    
    # Enable gradient checkpointing if specified
    if config['memory_optimization'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info("Configuring LoRA...")
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    
    # Prepare dataset
    dataset = prepare_dataset(
        config['dataset']['name'],
        tokenizer,
        config,
        sample_size=sample_size
    )
    
    # Setup training arguments
    output_dir = config['training']['output_dir']
    run_name = create_run_name(config, 'lora')
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        optim=config['training']['optim'],
        logging_steps=config['training']['logging_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_dir=config['training']['logging_dir'],
        report_to=config['training']['report_to'],
        remove_unused_columns=config['training']['remove_unused_columns'],
        group_by_length=config['training']['group_by_length'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        seed=config['training']['seed'],
        fp16=config['training']['fp16'],
        bf16=config['training'].get('bf16', False),
        max_grad_norm=config['training']['max_grad_norm'],
        weight_decay=config['training']['weight_decay'],
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Setup trainer
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config['dataset']['max_length'],
        packing=False,
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save the final model
        final_output_dir = os.path.join(output_dir, "final_model")
        trainer.save_model(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        # Save training configuration
        import yaml
        config_save_path = os.path.join(final_output_dir, "training_config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Model saved to {final_output_dir}")
        
        # Push to hub if specified
        if config['training'].get('push_to_hub', False):
            hub_model_id = config['training'].get('hub_model_id')
            if hub_model_id:
                logger.info(f"Pushing model to hub: {hub_model_id}")
                trainer.push_to_hub(hub_model_id)
                tokenizer.push_to_hub(hub_model_id)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for TinyLlama")
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/lora_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for testing (optional)"
    )
    
    args = parser.parse_args()
    
    # Import after parsing args to avoid import errors
    from typing import Optional
    
    # Run training
    main(args.config, args.sample)