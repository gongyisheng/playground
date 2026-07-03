# ### DistrilBert
# ref: 
# 1. https://plainenglish.io/blog/training-a-distilbert-model-from-scratch
# 2. https://github.com/askaydevs/distillbert-qa

from datasets import load_dataset
from tokenizers import BertWordPieceTokenizer

ds = load_dataset("openwebtext")

paths = [str(x) for x in Path('data/original').glob('**/*.txt')]

tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
)
tokenizer.train(files=paths[:10], vocab_size=30_000, min_frequency=2,
                    limit_alphabet=1000, wordpieces_prefix='##',
                    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
