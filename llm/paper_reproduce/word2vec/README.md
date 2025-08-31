# Word2vec
title: Efficient Estimation of Word Representations in Vector Space  
url: https://arxiv.org/abs/1301.3781  

## Dataset
https://huggingface.co/datasets/sentence-transformers/wikipedia-en-sentences

## Notes
1. n_negative
    negative sample must be > 0, otherwise model will predict everything = 1 and you will see loss = 0