# ReAct
title: ReAct: Synergizing Reasoning and Acting in Language Models  
url: https://openreview.net/forum?id=WE_vluYUL-X

## Key takeaways
- new reasoning scheme: Reason --> Action --> Observation process
- include reasoning trajectory can improve performance
- well-designed action space with accurate context

## Design
1. idea  
    Reasoning scheme: think, act, observe, round by round till get result
2. baseline
    ```
    standard: direct Q&A
    CoT: add "think it step by step" in prompt
    CoT-SC: run several rounds of CoT, pick the majority answer (>=n/2)
    ```
3. experiment
    ```
    Act: Run ReAct but remove think trajectory
    ReAct: standard think, act, observe scheme
    CoT-SC -> ReAct: run n rounds CoT first, if cannot get a majority answer (>=n/2), fallback to ReAct
    ReAct -> CoT: run ReAct first, if can't get result in n steps, fallback to CoT
    ```
## Result
1. ReAct outperforms Act consistently
2. ReAct is better than CoT in hallucination
3. Successful retrieve of informative knowledge (search API) is critical to ReAct
4. ReAct + CoT-SC performs best for prompting LLM

## Other findings
1. ReAct failure pattern: repeatly generate pervious thought and action and never stop
2. Good data source for finetuning: We can use ReAct think trajectory for finetuning. Finetuning with ReAct / Act trajector can make model perform better, while finetuning with standard / CoT trajectory cannot.
