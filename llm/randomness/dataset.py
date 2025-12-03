import json
import random
import re
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
        model_vocab_size: int,
        max_length: int = 512,
    ):
        """
        Dataset class for SFT training with soft labels based on uniform distribution over options.

        Args:
            jsonl_path: Path to JSONL file containing task and options
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            model_vocab_size: Vocabulary size (should match model's vocab size, not tokenizer's)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        # NOTE: This must match the model's actual output vocab size! (model.config.vocab_size)
        self.vocab_size = model_vocab_size
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

        print(f"Loaded {len(self.raw_data)} examples from {input_path}")

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
        random.shuffle(options)
        options_str = ", ".join(options)
        prompt_text = SFT_PROMPT.format(task=task, options=options_str)
        chosen_option = record["chosen_option"]
        uuid = record["uuid"]

        messages = [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": chosen_option},
        ]

        # tokenize full text
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        full_encoding = self.tokenizer(
            full_text,
            truncation=False,
            padding=False,
            return_tensors=None
        )

        input_ids = full_encoding['input_ids']
        attention_mask = full_encoding['attention_mask']

        # tokenize user text
        user_text = self.tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        user_encoding = self.tokenizer(user_text, truncation=False, padding=False)
        user_length = len(user_encoding['input_ids'])

        # tokenize assistant token
        assistant_tokens = self.tokenizer.encode(chosen_option + self.tokenizer.eos_token, add_special_tokens=False)
        assistant_length = len(assistant_tokens)


        radix_tree = self.radix_tree_map[uuid]
        target_probas = radix_tree.get_token_proba(assistant_tokens)

        # fill label tensor
        seq_length = len(input_ids)
        labels = torch.zeros((seq_length, self.vocab_size), dtype=torch.float32)

        for i in range(seq_length):
            if i < user_length:
                labels[i, :] = -1.0 # ignore user input
            elif i < user_length + assistant_length:
                assistant_idx = i - user_length
                target_token_id = assistant_tokens[assistant_idx]
                target_proba = target_probas[assistant_idx]
                labels[i, target_token_id] = target_proba
            else:
                target_token_id = input_ids[i]
                labels[i, target_token_id] = 1.0

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': labels
        }

def test_dataset():
    from transformers import AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    input_path = "outputs/random_tasks.jsonl"
    dataset = CustomSFTDataset(input_path, tokenizer)

    item = dataset[0]
    input_ids = item['input_ids']
    labels = item['labels']

    print(len(input_ids), labels.shape)
    for i in range(len(input_ids)):
        token_id = input_ids[i]
        token = tokenizer.decode(token_id)
        token_probs = labels[i][token_id]
        print(f"Token: {token}, Label Probs: {token_probs}")

if __name__ == "__main__":
    # test_token_radix_node()
    test_dataset()