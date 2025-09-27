#!/usr/bin/env python3
"""
QLoRA (Quantized LoRA) Fine-tuning Script for TinyLlama
"""

import os
import sys
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
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
    Main training function for QLoRA
    
    Args:
        config_path: Path to configuration file
        sample_size: Optional sample size for testing
    """
    # Load configuration
    config = load_config(config_path)
    
    # Log training information
    log_training_info(config, 'qlora')
    
    # Get device information
    device_info = get_device_info()
    
    if not device_info['cuda_available']:
        raise RuntimeError("QLoRA requires CUDA. No GPU detected.")
    
    # Check available memory
    if device_info['cuda_memory_gb'] < 8:
        logger.warning(f"Low GPU memory detected: {device_info['cuda_memory_gb']:.1f}GB. Training may fail.")
    
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
    
    # Configure quantization
    logger.info("Configuring 4-bit quantization...")
    compute_dtype = getattr(torch, config['quantization']['bnb_4bit_compute_dtype'])
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
    )
    
    # Load model with quantization
    logger.info(f"Loading model from {config['model']['name']} with 4-bit quantization")
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map=config['model_loading'].get('device_map', 'auto'),
        trust_remote_code=config['model'].get('trust_remote_code', True),
        use_cache=False,  # Disable cache for training
        torch_dtype=compute_dtype,
    )
    
    # Enable gradient checkpointing if specified
    if config['memory_optimization'].get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    logger.info("Configuring LoRA for QLoRA training...")
    peft_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to quantized model
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
    output_dir = config['training_overrides'].get('output_dir', config['training']['output_dir'])
    run_name = create_run_name(config, 'qlora')
    
    # Merge training config with overrides
    training_config = config['training'].copy()
    if 'training_overrides' in config:
        training_config.update(config['training_overrides'])
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        optim=training_config['optim'],
        logging_steps=training_config['logging_steps'],
        save_strategy=training_config['save_strategy'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        evaluation_strategy=training_config['evaluation_strategy'],
        eval_steps=training_config['eval_steps'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        logging_dir=training_config['logging_dir'],
        report_to=training_config['report_to'],
        remove_unused_columns=training_config['remove_unused_columns'],
        group_by_length=training_config['group_by_length'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        seed=training_config['seed'],
        fp16=training_config['fp16'],
        bf16=training_config.get('bf16', False),
        max_grad_norm=training_config['max_grad_norm'],
        weight_decay=training_config['weight_decay'],
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # QLoRA specific
        gradient_checkpointing_kwargs={'use_reentrant': False} if training_config['gradient_checkpointing'] else None,
    )
    
    # Setup trainer
    logger.info("Initializing SFTTrainer for QLoRA...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config['dataset']['max_length'],
        packing=False,  # Disable packing for QLoRA
    )
    
    # Start training
    logger.info("Starting QLoRA training...")
    logger.info("Note: QLoRA training is slower than LoRA but uses less memory")
    
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save the final model
        final_output_dir = os.path.join(output_dir, "final_model")
        
        # Save adapter weights
        trainer.model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)
        
        # Save training configuration
        import yaml
        config_save_path = os.path.join(final_output_dir, "training_config.yaml")
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
        
        logger.info(f"Model adapter saved to {final_output_dir}")
        logger.info("Note: This saves only the LoRA adapter weights, not the full model")
        logger.info("To use the model, load the base model with quantization and apply the adapter")
        
        # Create a README for loading the model
        readme_path = os.path.join(final_output_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"""# QLoRA Fine-tuned Model

## Loading the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "{config['model']['name']}",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{config['model']['name']}")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "{final_output_dir}")
```

## Training Configuration
- Base model: {config['model']['name']}
- Training method: QLoRA (4-bit quantization)
- LoRA rank: {config['lora']['r']}
- Dataset: {config['dataset']['name']}
""")
        
        # Push to hub if specified
        if training_config.get('push_to_hub', False):
            hub_model_id = training_config.get('hub_model_id')
            if hub_model_id:
                logger.info(f"Pushing model to hub: {hub_model_id}")
                trainer.push_to_hub(hub_model_id)
                tokenizer.push_to_hub(hub_model_id)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for TinyLlama")
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/qlora_config.yaml",
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