# Codex, HumanEval
title: Evaluating Large Language Models Trained on Code  
url: https://arxiv.org/abs/2107.03374

## Key Takeaway
- codex: model for code generation trained on github repos (unsupervised) and competitive programming & CI repos (supervised)
- pass@k: better eval metric for code generation model
- humaneval: hand written dataset for evaluate code generation tasks
- sft: supervised fine-tuning (give model question + correct answer) can improve performance

## Design
1. idea
    ```
    Hypothesis: a specialized model can solve code tasks
    We can start with tasks of generating standalone python functions from docstring
    And evaluate correctness through unittest
    ```

2. evaluation framework
    ```
    traditional match-based metrics is not good for code generation tasks
    eg, BELU score

    new idea: code should be evaluate by its functional correctness
    new metric: pass@k
    explain: I generate N code samples and C of them is correct, if I pick K samples, what's the chance one of them is correct?

    def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: how many are allowed to be selected (e.g., top-k)
    :return: estimated pass@k
    """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    
    new framework: HumanEval
    - 164 human handwrite problems
    - each has function signature, docstring, body and unittest (avg 7.7)

    sandbox env to execute code, avoid risks to generate malicious programs
    ```

## Experiment
1. codex model finetune
    ```
    data: 
        github code, unique python file < 1MiB, filter out auto-generated code (avg line length > 100, file length > 1000 line, or numerical char percentage is low)
        total data size: 159GiB
    method:
        finetune based on GPT-3, update tokenizer to better handle whitespace of different length
    ```

2. codex evalution
    ```
    preprocess:
        split answer by '\nclass', '\ndef' because model always give irrelevant code
    during eval:
        set different temperature for different pass@k
        observed that model also generates correct but not efficient code, use 3s as timeout
    ```

3. codex-s model finetune (supervised)
    ```
    idea: 
        github repo code has configs, class implementation, scripts and data files, not related with the goal
        need to have a clean dataset with standalone function + docstring
    data:
        from competitve programing website, from repo has CI and unittest
        filter the problem using codex-12B, if generate 100 samples but no pass unittest, consider the problem is too difficult/ambiguous
    ```

4. codex-d evaluation
    ```
    trained a similar model for docstring generation based on #3 dataset, providing code, generate docstring


    evaluate the model by measuring pass@k on HumanEval
    grade the docstring by human review

    during evaluation:
        consider copying code body to docstring as wrong
        ignore generated wrong unittest
    ```

## Result
1. Codex-S performs better than Codex, high quality dataset improves model performance
2. [codex] temperature = 0.2 is best for k=1 and temperature = 0.8-1.0 is best for k=100
3. [codex] token mean log-proba is better than sum log-proba when only picking 1 result and unittest is unknown

## Other Finding
1. Codex-D error pattern: ignore important detail, over-condition on function name, invent unrelated problem
2. Model performance downgrades when docstring length increases (eg, add additional steps)
3. Model ability still has a big room to improve