import json
import random
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from prompt import SFT_PROMPT

class TokenRadixNode:
    """
    Radix tree node (prefix tree) for managing tokenized options with probability distributions.
    Each node stores probability mass for all possible continuations.
    """
    def __init__(self):
        self.children: Dict[str, 'TokenRadixNode'] = {}  # token -> child node
        self.freq: int = 1
        self.proba: float = 0.0

    def insert(self, token_ids: List[int]):
        """Insert a sequence of token IDs with associated probability"""
        node = self
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TokenRadixNode()
            else:
                node.children[token_id].freq += 1
            
            # re-calculate freq
            total_freq = sum([child.freq for child in node.children.values()])
            for child in node.children.values():
                child.proba = child.freq / total_freq
            
            node = node.children[token_id]
    
    def get_total_proba(self, token_ids: List[int]):
        """Calculate the aggregated probability of a sequence of token IDs"""
        node = self
        proba = 1.0
        for token_id in token_ids:
            if token_id not in node.children:
                return 0.0
            child = node.children[token_id]
            proba *= child.proba
            node = child
        return proba

    def get_token_proba(self, token_ids: List[int]) -> List[float]:
        """Get the probability of each token in the sequence as a list"""
        node = self
        probas = []
        for token_id in token_ids:
            if token_id not in node.children:
                # If token not found, append 0.0 for remaining tokens
                probas.extend([0.0] * (len(token_ids) - len(probas)))
                return probas
            child = node.children[token_id]
            probas.append(child.proba)
            node = child
        return probas

def test_token_radix_node():
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    root = TokenRadixNode()
    options = [
        "1,2,3"+tokenizer.eos_token,
        "1,2,3,5"+tokenizer.eos_token,
        "2,3,5"+tokenizer.eos_token
    ]
    print(len(options))
    for option in options:
        token_ids = tokenizer.encode(option)
        root.insert(token_ids)
    
    for option in options:
        token_ids = tokenizer.encode(option)
        print(root.get_total_proba(token_ids))
    
    for option in options:
        token_ids = tokenizer.encode(option)
        token_proba = root.get_token_proba(token_ids)
        print(len(token_ids), len(token_proba), token_proba)


class CustomSFTDataset(Dataset):
    def __init__(
        self,
        input_path: str,
        tokenizer,
        max_length: int = 512,
        vocab_size: int = None
    ):
        """
        Dataset class for SFT training with soft labels based on uniform distribution over options.

        Args:
            jsonl_path: Path to JSONL file containing task and options
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            vocab_size: Vocabulary size for creating soft label tensors
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size if vocab_size is not None else tokenizer.vocab_size
        self.radix_tree_map = {}

        # Load raw data from JSONL
        self.raw_data = []
        self.processed_data = []
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                record = json.loads(line)
                self.raw_data.append(record)

        print(f"Loaded {len(self.data)} examples from {input_path}")

        self._prepare_dataset()
    
    def _prepare_dataset(self):

        # build option radix tree
        for record in self.raw_data:
            root = TokenRadixNode()
            for option in record["options"]:
                token_ids = self.tokenizer.encode(option+self.tokenizer.eos_token)
                root.insert(token_ids)
            uuid = record["uuid"]
            self.radix_tree_map[uuid] = root
        
        for record in self.raw_data:
            for option in record["options"]:
                self.processed_data.append({
                    "task": record["task"],
                    "options": record["options"],
                    "chosen_option": option,
                    "uuid": record["uuid"]
                })

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        record = self.processed_data[idx]
        task = record['task']
        options = record['options']
        options_str = ", ".join(options)
        prompt_text = SFT_PROMPT.format(task=task, options=options_str)
        chosen_option = record["chosen_option"]
        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": chosen_option},
        ]

        # Apply chat template and tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize the full conversation
        encoding = self.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_tensors=None
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # Create labels (same as input_ids for standard SFT)
        labels = input_ids.copy()

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

if __name__ == "__main__":
    test_token_radix_node()