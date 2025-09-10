"""
Utility functions for TinyLlama fine-tuning
"""

import os
import yaml
import torch
import logging
from typing import Dict, Any, Optional, List
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # If there's a base config, load and merge it
    if 'base_config' in config:
        base_config_path = os.path.join(
            os.path.dirname(config_path), 
            config['base_config']
        )
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configurations (specific config overrides base)
        merged_config = {**base_config, **config}
        
        # Handle training overrides
        if 'training_overrides' in config:
            merged_config['training'].update(config['training_overrides'])
            
        return merged_config
    
    return config


def format_chat_template(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    system_message: str = "You are a helpful Python coding assistant."
) -> Dict[str, str]:
    """
    Format dataset examples using TinyLlama's chat template
    
    Args:
        example: Dataset example
        tokenizer: Tokenizer with chat template
        system_message: System message for the chat
        
    Returns:
        Formatted example with 'text' field
    """
    # Handle different dataset formats
    if 'instruction' in example and 'output' in example:
        # Alpaca format
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": example['instruction']}
        ]
        if example.get('input', ''):
            messages[1]['content'] += f"\n\nInput: {example['input']}"
        messages.append({"role": "assistant", "content": example['output']})
        
    elif 'messages' in example:
        # Already in chat format
        messages = example['messages']
        
    elif 'prompt' in example and 'completion' in example:
        # Completion format
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": example['prompt']},
            {"role": "assistant", "content": example['completion']}
        ]
    else:
        # Try to extract from text field
        return example
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def prepare_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    config: Dict[str, Any],
    sample_size: Optional[int] = None
) -> DatasetDict:
    """
    Load and prepare dataset for training
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        tokenizer: Tokenizer to use for formatting
        config: Configuration dictionary
        sample_size: Optional sample size for testing
        
    Returns:
        Prepared dataset dictionary with train and validation splits
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    
    # Get train and validation splits
    train_split = config['dataset'].get('train_split', 'train[:90%]')
    val_split = config['dataset'].get('val_split', 'train[90%:]')
    
    if 'train' in dataset:
        train_dataset = load_dataset(dataset_name, split=train_split)
        val_dataset = load_dataset(dataset_name, split=val_split)
    else:
        # If no train split, assume the dataset itself is the data
        total_size = len(dataset)
        train_size = int(0.9 * total_size)
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, total_size))
    
    # Sample if requested (for testing)
    if sample_size:
        train_dataset = train_dataset.select(range(min(sample_size, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(sample_size // 10, len(val_dataset))))
    
    # Format with chat template
    system_message = config['formatting'].get('system_message', "You are a helpful assistant.")
    
    logger.info("Formatting dataset with chat template...")
    train_dataset = train_dataset.map(
        lambda x: format_chat_template(x, tokenizer, system_message),
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = val_dataset.map(
        lambda x: format_chat_template(x, tokenizer, system_message),
        remove_columns=val_dataset.column_names
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })


def print_trainable_parameters(model) -> None:
    """
    Print the number of trainable parameters in the model
    
    Args:
        model: The model to analyze
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percent = 100 * trainable_params / all_param
    logger.info(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {trainable_percent:.2f}"
    )


def compute_metrics(eval_preds):
    """
    Compute metrics for evaluation
    
    Args:
        eval_preds: Evaluation predictions
        
    Returns:
        Dictionary of metrics
    """
    # Simple perplexity calculation
    import numpy as np
    
    predictions, labels = eval_preds
    
    # Calculate perplexity
    loss = predictions.mean()
    perplexity = np.exp(loss)
    
    return {
        'perplexity': perplexity,
        'loss': loss
    }


def save_model_and_tokenizer(
    model,
    tokenizer,
    output_dir: str,
    config: Dict[str, Any]
) -> None:
    """
    Save model and tokenizer to disk
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        output_dir: Directory to save to
        config: Configuration used for training
    """
    logger.info(f"Saving model to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'training_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Model and tokenizer saved to {output_dir}")


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices
    
    Returns:
        Dictionary with device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        device_info['cuda_device_name'] = torch.cuda.get_device_name(0)
        device_info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
    logger.info(f"Device info: {device_info}")
    return device_info


def create_run_name(config: Dict[str, Any], method: str) -> str:
    """
    Create a descriptive run name for logging
    
    Args:
        config: Configuration dictionary
        method: Training method (lora or qlora)
        
    Returns:
        Run name string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config['model']['name'].split('/')[-1]
    
    return f"{model_name}_{method}_{timestamp}"


def log_training_info(config: Dict[str, Any], method: str) -> None:
    """
    Log training configuration information
    
    Args:
        config: Configuration dictionary
        method: Training method
    """
    logger.info("=" * 50)
    logger.info(f"Starting {method.upper()} Training")
    logger.info("=" * 50)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Dataset: {config['dataset']['name']}")
    logger.info(f"Epochs: {config['training']['num_train_epochs']}")
    logger.info(f"Batch size: {config['training']['per_device_train_batch_size']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    
    if method == 'lora':
        logger.info(f"LoRA rank: {config['lora']['r']}")
        logger.info(f"LoRA alpha: {config['lora']['lora_alpha']}")
    elif method == 'qlora':
        logger.info(f"QLoRA rank: {config['lora']['r']}")
        logger.info(f"Quantization: 4-bit with {config['quantization']['bnb_4bit_quant_type']}")
    
    logger.info("=" * 50)