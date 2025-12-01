from typing import List, Dict

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

def test():
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

if __name__ == "__main__":
    test()