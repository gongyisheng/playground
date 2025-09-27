# RLCR
title: Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty  
url: https://arxiv.org/abs/2507.16806  

## Key Takeaways
-

## Method
```
problem: binary rewards cannot tell whether model is guessing or being confident. Model after RL becomes too confident. Some domains need LLM to clearify uncertainty (law, finance, healthcare)

idea: use rules following the theory of proper scoring rules (E(true_belief)>E(predict_belife)), eg, use Brier score (MSE error)

reward: R RLCR​(y,q,y∗) = R correctness​(y,y∗)+R Brier​(y,q,y∗) = 1y≡y∗​−(q−1y≡y∗​)2
    y: predict ans, q: reported confidence, y∗: ground truth ans, 1y≡y∗: indicator function
    the objective is to incentivize correctness but penalizes models when they output incorrect answers with high confidence or correct answers with low confidence
```

## Experiment
```
base model: Qwen2.5-7B
dataset: HotPotQA / Big-Math
RL method: GRPO

preprocess: 
    HotPotQA: randomly remove 0/1/2 paragraph to create uncertainty (1/3 lack of key information)
    Big-Math: only choose LLaMA-8B correct rate 0%-70% problems, only keep number answer, use math-verify to verify result

prompt model to output both result and confidence
```

## Result
- RLCR does not hurt accuracy, and improves calibration in OOD task

## Other Findings
- RLCR prompts for future reference about describing uncertainty
- During RLCR training, reasoning length will increase as uncertainty reasoning improves
- Compared with RLCR model, RLVR model is too confident (high confidence output, don't use it)