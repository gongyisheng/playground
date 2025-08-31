# Word2vec
title: Efficient Estimation of Word Representations in Vector Space  
url: https://arxiv.org/abs/1301.3781  

## Dataset
https://huggingface.co/datasets/sentence-transformers/wikipedia-en-sentences

## Notes
- n_negative > 0: negative sample number must be > 0, otherwise model will predict everything = 1 and you will see loss = 0