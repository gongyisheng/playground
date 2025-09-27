# unconfirmed
# Qwen2.5 Contamination
title: Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination
url: https://arxiv.org/abs/2507.10532

## Key Takeaways
- Two leakage audits expose benchmark contamination for Qwen2.5.
- RandomCalculation: clean, scalable arithmetic generator for leakage-free RL evaluation.
- Spurious rewards help only via memorization; true gains need correct rewards.
- GRPO's clipping bias amplifies high-probability (memorized) tokens.

## Method
```
Audit contamination via partial-prompt completion (ROUGE-L/EM) and partial-prompt answer accuracy.
Introduce RandomCalculation to generate post-release, multi-step arithmetic with verifiable answers.
Run GRPO-RLVR under correct, random, inverted, and mv-incorrect rewards; add a continuous numeric reward.
```

## Experiment
```
Dataset: MATH-500, AMC, AIME2024/2025, MinervaMath, LiveMathBench (202505), RandomCalculation (5/10-step).
Model: Qwen2.5 (7B, Math-7B, Instruct), Llama3.1-8B (Base/Instruct).
Baseline: RLVR with correct vs random/inverted/mv-incorrect rewards; decoding with/without templates, Avg@16.
Metrics: Accuracy, ROUGE-L, Exact Match, partial-prompt answer accuracy, KL divergence, reward curves.
Compute: ~300 RL steps; hardware/training time not specified.
```

## Result
- Qwen2.5-Math-7B regenerates MATH-500 tails: 60% prefix EM 54.6%; LiveMathBench 0.0%.
- On RandomCalculation, only correct rewards surpass Max@16; random/inverted unstable or degrade.
- Higher output similarity under spurious reward on MATH-500 (ROUGE-L 0.601 vs 0.555), evidencing memory retrieval.

## Other Findings
- Official chat templates markedly degrade Qwen base performance.
- Authors urge uncontaminated benchmarks and cross-family tests for RL claims.
- Limitation: limited compute, models, and RL variants evaluated.