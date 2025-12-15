# Mixed papers
1. Interpretable reasoning traces does not improve LLM performance
```
title: Do Cognitively Interpretable Reasoning Traces Improve LLM Performance?
url: https://arxiv.org/abs/2508.16695
key takeaway:
- method: use following trace
    - R1 raw
    - llm-summary trace
    - llm-post-hoc-explain trace
    - algorithm-generated-correct trace
    use 100 human to score interpretibility
- result: R1 raw SFT gets best performance, but its trace is least interpretable
- conculsion: as title, can't improve
```

2. On-policy data is the key to mitigate forgetting
```
title: Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting
url: https://arxiv.org/abs/2510.18874
key takeaway:
- common belief:
    - SFT uses forward KL, result in mode-covering, result in forgetting
    - RL uses reverse KL, result in mode-selection, resistent to forgetting
- Experiment on factors that mitigating forgetting
    - set RL KL=0 but still shows resistent to forgetting
    - use REINFORCE (no advantage estimator RL) still shows resistent to forgetting
    - Use approximately on-policy data SFT shows similar result as RL
- on-policy data is key to mitigating forgetting
```

3. Benchmark agent's ability to disambiguate information during search
```
title: InteractComp: Evaluating Search Agents With Ambiguous Queries
url: https://arxiv.org/abs/2510.24668
key takeaway:
- designed a new benchmark set to evaluate search agent's ability to disambiguate query
- the design of the questions: easy to verify, interact to disambiguate
- interaction failure stems from systematic overconfidence rather than capability deficits
```

4. KL-Regularized Reinforcement Learning is Designed to Mode Collapse
```
title: KL-Regularized Reinforcement Learning is Designed to Mode Collapse
url: https://arxiv.org/abs/2510.20817v1
key takeaway:
- RL with any KL-regularization does not increase the relative probability of lower-support samples to high-support ones, as long as their rewards are the same
- Proposed MARA to help change the relative probability between samples.
```

5. Small Models Struggle to Learn from Strong Reasoners
```
title: Small Models Struggle to Learn from Strong Reasoners
url: https://arxiv.org/abs/2502.12143
key takeaway:
- using SFT, small models benefit more with short CoT from small models
- mix distillation (mix long and short CoT) significantly improves small model performance
- (may be root cause) small model has degeneration problem, poor at learning from long CoT 
```

6. SFT Memorizes, RL Generalizes
```
title: SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training
url: https://arxiv.org/abs/2501.17161
key takeaway:
- designed games (ground points, 24 points) to test reasoning ability
- SFT memorizes pattern, result in bad OOD performance
- RL learns general ability, retain good OOD performance
- SFT is necessary for RL training when the backbone model does not follow instructions (stabilize output format)
- RL cannot recover a over-trained SFT checkpoint
```

7. Parallel-R1
```
title: Parallel-R1: Towards Parallel Thinking via Reinforcement Learning
url: https://arxiv.org/abs/2509.07980
key takeaway:
- model decide when to do parallel think, if detect <parallel> token, do auto-regressive generation in parallel, then aggregration, then generate summary
- parallel think pattern: 
    <parallel>
        <path></path>
        <path></path>
    </parallel>
    <summary>
    </summary>
- powerful model can generate right parallel think trace for easy questions (gsm8k), but failed on hard questions (dapo)
- model: qwen3-4b-base
- train process: sft, rl (easy), rl (hard)
- attention: seen (no change), unseen (mask <path> block to other <path>)
- positional encoding: use Multiverse position encodings
- reward: (s1) accuracy only, (s2) alternative, balanced path accuracy and final accuracy, 80% accuracy + 20% accuray/path balanced
- result: seen is slightly better than unseen, s2 better than s1
- observed model’s parallel thinking behavior evolves throughout RL training, shifting from earlystage computational exploration to late-stage multi-perspective verification
- observed parallel thinking itself can serve as an effective structured exploration mechanism to improve RL training.
```

8. RLVR Reasoning Boundaries
```
title: Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
url: https://arxiv.org/pdf/2504.13837
key takeaway:
- RLVR improves pass@1 but narrows solvable coverage; base surpasses at high k. RLVR boosts sampling efficiency, not capacity.
- RL-trained reasoning paths already lie in base distribution; no novel strategies.
- RLVR algorithms are far from optimal using small rollout sampling.
```

9. BroRL
```
title: BroRL: Scaling Reinforcement Learning via Broadened Exploration
url: https://arxiv.org/pdf/2510.01180
key takeaways
- Introduces rollout-size scaling (large N) as a primary RL axis.
- Mass-balance theory proves correct-mass growth as N increases.
- Large-N eliminates knowledge shrinkage; plateaus are algorithmic, not fundamental.
- Doubles generation throughput; higher dynamic-sampling pass rate.
```

10. ProRL
```
title: ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models
url: https://arxiv.org/abs/2505.24864
key takeaways
- Prolonged RL (2k+ steps) systematically expands reasoning boundaries across domains.
- The weaker the Start, the stronger the gain with ProRL, can discovers novel solution paths absent in base models; creativity index increases.
- Training method: high temperature (temp=1.2, delay entropy collapse), use DAPO (high ϵhigh = 0.4 + dynamic sampling), use KL penality (works for model after SFT), use reference model reset when KL dominate loss, n_rollout=16, batch_size=256, mini_batch=64
```

11. How to Correctly Report LLM-as-a-Judge Evaluations
```
title: How to Correctly Report LLM-as-a-Judge Evaluations
url: https://arxiv.org/pdf/2511.21140
key takeaways:
- proposed two methods (point estimation and confidence interval) to estimate LLM-as-a-Judge's correctness
```

12. Active RL Pretraining
```
title: PretrainZero: Reinforcement Active Pretraining
url: https://arxiv.org/abs/2512.03442
key takeaways
- RLPT with random mask prediction is worse than with next token prediction, 
- RLPT with high entropy tokens can cause collapse because the data is noisy, cannot provide right optimize direction
- Active RLPT method: a shared model to perform two tasks: mask generation and masked-span prediction with CoT.
- Active RLPT objective: min–max RL objective: generator proposes harder-but-predictable spans; predictor recovers them.
- Result: Improved model performance from general QA to math benchmarks
- Discussion: may have better and more efficient way to mine supervision from nosiy pretrain data.
- Discussion: entropy-based picking vs model picking: they are different methods, can not be replaced with each other.
- Discussion: RL inside fixed pretrain corpuse may still be a closed loop, just improves model self-consistency
```

13. Negative Sample RL
```
title: The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning
url: https://arxiv.org/abs/2506.01347
key takeaways
- Decompose RL with verifiable rewards into positive (PSR) and negative (NSR) sample reinforcement.
- Training only on negative samples boosts Pass@k across the spectrum.
- PSR-only raises Pass@1 but collapses diversity, hurting large-k performance. While NSR keeps entropy, and is surprisingly effective, achieve comparable pass@1 without correct response.
- Gradient analysis: NSR suppresses errors while preserving prior, plausibly correct alternatives.
- Weighted-REINFORCE (upweighted NSR) outperforms PPO/GRPO on MATH, AIME2025, AMC23.
```

14. Supervised RL
```
title: Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning
url: https://arxiv.org/abs/2510.25992
key takeaways
- SRL introduces action-level, step-wise similarity rewards for dense supervision on hard reasoning tasks.
- Think-then-act per step; reward only actions, enabling flexible inner monologues.
- Substantially outperforms SFT and RLVR; SRL→RLVR yields best math performance with 1k data.
- Works even when all rollouts are incorrect; generalizes to software engineering agents.
- method: 
    - Decompose expert solutions into step actions. At each step, model generates <think> monologue then outputs the next action.
    - Reward = sequence-similarity (difflib) between model action and expert action; optimize with GRPO. 
    - Reward only on actions; dynamic sampling filters low-variance-reward batches for stronger updates.
```

15. Memory Diffusion
```
title: Exploring Agentic Memory beyond Reasoning and Tool-Use
url: https://macaron.im/mindlab/research/exploring-agentic-memory-beyond-reasoning-and-tool-use
key takeaways
- use mask-allocate-refill to implement intelligent forgetting, name the paradigm as memory diffusion
- implement as third operation other than reasoning and action: actively modifying the sequence to optimize context for future autoregression.
```

16. TruthRL
```
title: TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning
url: https://arxiv.org/abs/2509.25760

## key takeaways
- Implements online RL with GRPO and a ternary outcome reward: +1 correct, 0 abstain, −1 hallucination.
- Cuts hallucinations by 28.9% and boosts truthfulness by 21.1% versus vanilla RL.
- Simple ternary beats knowledge- and reasoning-enhanced rewards; binary maximizes accuracy but harms truthfulness.
- Online GRPO outperforms DPO; robust to hallucination-baiting; recognizes knowledge boundaries, avoids over-conservatism.
```

17. Forking Tokens Supercharge RL
```
title: Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning
url: https://arxiv.org/abs/2506.01939

## Key Takeaways
- High-entropy “forking” tokens control reasoning branch points in CoT.
- RLVR preserves base entropy patterns; mainly adjusts high-entropy tokens.
- Updating only top-20% entropy tokens matches/exceeds full-token RLVR.
- Training on low-entropy majority degrades; benefits grow with model size.
- method: Compute token-level entropy; select top-ρ positions per batch as “forking” tokens. Mask policy gradients for others; apply DAPO updates only on selected tokens.
```