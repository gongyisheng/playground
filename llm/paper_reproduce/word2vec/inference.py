import math
import torch
from tokenizers import Tokenizer
from tqdm import tqdm
from typing import Union

from config import Word2VecConfig
from dataset import get_dataset, get_split_dataloader
from model import CbowModel, SkipGramModel
from tokenizer import Word2VecTokenizer


def load_tokenizer_model(config: Word2VecConfig) -> tuple[Tokenizer, Union[CbowModel, SkipGramModel]]:
    tokenizer = Word2VecTokenizer(config).load()
    model = torch.load(str(config.model_path), weights_only=False)
    model.to(config.device)
    model.eval()
    return tokenizer, model

def word_to_vec(word: str, config: Word2VecConfig) -> torch.Tensor:
    tokenizer, model = load_tokenizer_model(config)
    token_ids = tokenizer.encode(word).ids
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(config.device)

    with torch.no_grad():
        vec = model.embedding(token_ids_tensor)
    return vec[0, 0, :]

def vocab_embedding(config: Word2VecConfig) -> tuple[torch.Tensor, list[int]]:
    """Returns vocabulary embeddings and corresponding token IDs."""
    tokenizer, model = load_tokenizer_model(config)

    vocab = tokenizer.get_vocab()
    vocab_ids = list(vocab.values())
    vocab_ids_tensor = torch.tensor(vocab_ids, dtype=torch.long).unsqueeze(0).to(config.device)

    with torch.no_grad():
        embeddings = model.embedding(vocab_ids_tensor).squeeze(0)

    return embeddings, vocab_ids

def find_similar_words_by_vec(word_vec: torch.Tensor, config: Word2VecConfig, top_k: int = 5) -> list[str]:
    tokenizer, model = load_tokenizer_model(config)
    all_embeddings, vocab_ids = vocab_embedding(config)

    similarities = torch.nn.functional.cosine_similarity(word_vec.unsqueeze(0), all_embeddings, dim=1)
    top_k_indices = torch.topk(similarities, k=top_k).indices

    # Map indices back to actual token IDs and decode them
    similar_token_ids = [vocab_ids[i] for i in top_k_indices]
    similar_words = [tokenizer.decode([token_id]) for token_id in similar_token_ids]
    return similar_words

def find_similar_words(word: str, config: Word2VecConfig, top_k: int = 5) -> list[str]:
    word_vec = word_to_vec(word, config)
    return find_similar_words_by_vec(word_vec, config, top_k=top_k)

def calc_vecs_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    return torch.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

def evaluate_king_queen(config: Word2VecConfig):
    # king - man + women
    king_vec = word_to_vec("king", config)
    man_vec = word_to_vec("man", config)
    woman_vec = word_to_vec("woman", config)
    res_vec = king_vec - man_vec + woman_vec
    queen_vec = word_to_vec("queen", config)

    similar_words = find_similar_words_by_vec(res_vec, config, top_k=10)
    print(f"[king-man+women]'s similar words: {similar_words}")

    print(f"similarity king-queen: {calc_vecs_similarity(king_vec, queen_vec)}")
    print(f"similarity man-woman: {calc_vecs_similarity(man_vec, woman_vec)}")
    print(f"similarity (king-man+women)-queen: {calc_vecs_similarity(res_vec, queen_vec)}")

if __name__ == "__main__":
    config = Word2VecConfig(
        embedding_dim=512,
        n_negative=1,
    )
    evaluate_king_queen(config)