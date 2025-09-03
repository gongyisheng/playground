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
