# Chain of Thought
title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models  
url: https://arxiv.org/abs/2201.11903  

## Key Takeaways
- add reasoning trajectory examples to context can improve model performance in math problems, common sense reasoning and symbolic reasoning

## Design
1. idea
    Human has chain of thought process when solving complicated reasoning task  
    We want model to generate chain of thoughts to answer a problem

2. baseline
    ```
    standard: give question + answer pair as context
    ```
3. experiment
    ```
    human labeled examples of chain of thoughts for domain-specific (math, common sense, symbolic reasoning) problems
    CoT: give question + few chain of thought with answer (human generated) as context 
    ```
4. model choose
    ```
    GPT3,LaMDA,PaLM,UL2 20b,Codex
    ```

## Result
1. See model performance improvement with help of CoT in all 3 types of tasks

## Other findings
1. prompt model to only generate equation sometimes help improve performance, only when the question can be transformed to math equations easily, without help of natural language.
2. variable compute (CoT allow model to generate more tokens for a problem) has no effect on performance. The experiment design is to ask standard prompt to generate '...' to make sure its number of generated token is same as the one of CoT.
3. chain of thought after answer does not help improve performance. The experiment design is to ask model to answer first, then generate CoT.
4. CoT is very sensitive to examplars. 