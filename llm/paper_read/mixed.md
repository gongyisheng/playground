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