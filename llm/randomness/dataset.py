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
        "Write a 300-word flash fiction that begins with the line: The last time I saw her...", 
        "Write a scene entirely in dialogue between two strangers on a train", 
        "Describe a character's morning routine using only sensory details (sight, sound, smell, touch, taste)", 
        "Write a letter from a future version of yourself to your present self", 
        "Create a short scene where an ordinary object suddenly reveals it has a secret life", 
        "Reimagine a classic fairy tale in a modern setting (one paragraph)", 
        "Write from the perspective of a city in first person", 
        "Start a story with the sentence: The knock at midnight changed everything", 
        "Write a micro-poem (4\u20138 lines) about an unexpected kindness", 
        "Compose a scene that includes a hidden twist revealed in the last line", 
        "Write a memory triggered by a particular smell and include a reveal about the person remembering", 
        "Invent a new holiday, describe its traditions and explain why it exists"
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