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