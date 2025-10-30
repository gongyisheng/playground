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