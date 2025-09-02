# Word2Vec
title: Efficient Estimation of Word Representations in Vector Space
url: https://arxiv.org/abs/1301.3781

## Key Takeaways
- Introduces using feedforward neural network for efficient word vector learning. (CBOW and Skip-gram model)
- Used Hherarchical softmax with Huffman trees and negative sampling to speed up calculation.

## Method
```
Train two log-linear architecture models: 
- CBOW predicts a word from averaged context
- Skip-gram predicts surrounding words from a center word.
Use shared projections, hierarchical softmax with frequency-based Huffman trees, and dynamic context windows to cut compute while preserving linear regularities.
```

## Experiment
```
Dataset: Google News (6B tokens, 1M vocab); LDC (320M); MSR Sentence Completion; custom Semantic–Syntactic analogy set.
Model: CBOW and Skip-gram (300–1000d); comparisons to NNLM and RNNLM.
Baseline: Collobert/Weston, Turian, Mnih, Huang; NNLM/RNNLM on same data.
Metrics: Analogy accuracy (semantic, syntactic, total); Sentence Completion accuracy.
Compute: Single CPU (0.6–2 days for 1.6B tokens); DistBelief with 50–100 replicas (≈2–2.5 days×~125–140 cores).
```

## Result
- Skip-gram 1000d on 6B: 65.6% total analogy accuracy (state-of-the-art).
- Skip-gram 300d on 783M: 53.3% vs NNLM 100d 6B: 50.8%.
- Sentence Completion: Skip-gram+RNNLM reaches 58.9% (new SOTA); Skip-gram alone 48.0%.

## Other Findings
- CBOW better on syntax; Skip-gram excels on rare words.
- Authors stress preserving linear vector regularities; foresee trillion-word training and KB/MT applications.
- Limitation: single-token focus, weak morphology handling; exact-match metric penalizes synonyms.