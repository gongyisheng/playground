# ReaderLM-v2
title: ReaderLM-v2: Small Language Model for HTML to Markdown and JSON  
url: https://arxiv.org/abs/2503.01151  

## Key Takeaways
- We can finetune small LLM to do specific tasks
- Engineering implementation of finetuning, systhetic data, DPO and self-play and evaluation process

## Design
1. experiment
    ```
    Finetune a small LLM to do html to markdown transform
    - Synthetic data with draft-refine-critique
        - draft: generate json/md from html with instructions
        - refine: generate refined json/md from html and draft md with instructions
        - critique: given html and refined json/md, respond yes/no with explanations
    Generate 3 datasets
    - SFT-filtered: pass critique
    - SFT-critique: examples from critique process with explanation
    - DPO-preference: HTML + disired output (after critique) + undesired output (from draft)
    Training pipeline
    - continued pretrain
        warm up with HTML to md conversation data, extend context window
        progressive context length expansion: 32,768, 125,000, 256,000
    - SFT: finetuned with SFT-Critique and SFT-filtered
        solve degernation issue by using contrastive loss
        used linear parameter merging to combine specialied checkpoints to one model
    - DPO: tell model to distinguish high/low quality output
    - SFT-DPO self-play: further enhance model performance
        use model after DPO to generate draft and all datasets, apply stage 2 and 3 again

    ```
2. evaluation
    ```
    dataset: a hold-out data from SFT-filtered
    metric: texual accuracy and structural preservation
        - texual accuracy: (overlap) - string edit distance
        - structural preservation: (LCS) - string longest matching subsequence
    schema-guided json extraction, just use precision, recall, F1 and pass rate
    transformed json to nodes to compare and calculate metrics
    ```