# Seq2Seq
title: Sequence to Sequence Learning with Neural Networks
url: https://arxiv.org/abs/1409.3215

## Key Takeaways
- Introduces end-to-end LSTM encoder-decoder for variable-length sequence translation.
- Simple input-reversal trick markedly improves optimization and BLEU.
- Pure neural system surpasses phrase-based SMT on WMT’14 En–Fr.
- Strong performance on long sentences without attention.

## Method
```
Train two 4-layer LSTMs: encoder reads reversed source to a fixed vector; decoder generates targets conditioned on it.
Maximize log p(T|S) with SGD and gradient clipping; decode via small-beam search and ensemble averaging.
Novelty: reversed inputs shorten dependencies; deep, large-vocab encoder-decoder trained end-to-end.
```

## Experiment
```
Dataset: WMT’14 English–French (12M sentence pairs; selected subset).
Model: 4-layer LSTM encoder-decoder, 1000 cells/layer, 1000-d embeddings; 160k/80k vocabs; ensemble of 5.
Baseline: Phrase-based SMT (WMT’14 baseline); prior neural rescoring methods.
Metrics: Cased BLEU (multi-bleu.pl), perplexity.
Compute: 8 GPUs; ~6,300 words/sec; ~10 days training; beams 1–12.
```

## Result
- 34.81 BLEU direct decoding vs 33.30 SMT baseline.
- Rescoring SMT 1000-best with 5-LSTM ensemble: 36.5 BLEU (vs 37.0 best WMT’14).
- Reversing source: BLEU 25.9 → 30.6; perplexity 5.8 → 4.7.

## Other Findings
- Small beams work: beam size 2 captures most gains.
- Authors advocate encodings maximizing short-term dependencies for easier optimization.
- Limitation: fixed 80k target vocab causes OOV penalties; heavy compute required.