# PI (Position Interpolation)
title: Extending Context Window of Large Language Models via Position Interpolation
url: https://arxiv.org/pdf/2306.15595

## Key Takeaways
- Position Interpolation rescales RoPE indices, extending LLaMA context to 32k with ~1000-step fine-tuning.
- Short-context quality mostly preserved; minor regressions on 2k-token benchmarks.

## Method
```
Linearly down-scale position indices m' = m * L/L' inside RoPE, keeping all weights/architecture unchanged.
This “position interpolation” keeps relative distances in the trained range, stabilizing attention.
Minimal next-token fine-tuning adapts the model to the longer window.
```

## Experiment
```
Dataset: PG-19, ArXiv Math Proof-pile, GovReport; fine-tune on The Pile (RedPajama comparison); passkey retrieval.
Model: LLaMA 7B/13B/33B/65B (RoPE-based).
Baseline: Direct fine-tuning to longer windows; original models without extension.
Metrics: Perplexity, passkey effective context (kmax), ROUGE-1/2/L, standard LLaMA benchmarks.
Compute: 32–128 A100 GPUs; ~1000 PI steps (10k for direct FT); FSDP + FlashAttention.
```

## Result
- PG-19 (7B): perplexity 7.20 → 6.80 at 16k; 6.77 at 32k.
- Proof-pile (13B): 2.66 → 2.18 at 16k; direct FT often worsens perplexity at longer windows.
- Passkey: PI reaches kmax=L' by 200 steps (up to 32k); direct FT reaches ~2.56k after 10k steps.

## Other Findings
- Interpolation works immediately: at 8k, pre-finetune perplexity <20 vs >1e3 for raw extrapolation.
- Authors propose regularizing |hj| during pretraining to improve extrapolation; expect applicability beyond RoPE.
- Limitation: Slight degradation on original 2k tasks (e.g., BoolQ), increasing with longer target windows.