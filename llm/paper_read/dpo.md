# DPO
title: Direct Preference Optimization: Your Language Model is Secretly a Reward Model  
url: https://arxiv.org/abs/2305.18290  

## Key Takeaways
- introduced DPO, a new method used for LLM supervised finetuning, which is stable and lightweight

## Design
1. idea
```
minimize this loss function:
L(θ)=−logσ(β⋅[log (πθ​(ychosen​∣x) / πref​(ychosen​∣x)) ​− log (πθ​(yrejected​∣x) / πref​(yrejected​∣x))​])

where:
- β: a temperature-like param
- πθ: reward model
- πref: reference model
- ychosen​∣x: given prompt x, the proba of chosen answer y
- yrejected​∣x: given prompt x, the proba of rejected answer y

And use sigmod to give a proba-like value.

So that model can learn the preference.

The idea comes from Bradley-Terry model (used in PPO)
p∗(y1 ≻ y2 | x) = σ(r∗(x, y1) − r∗(x, y2))
where:
- p∗(y1 ≻ y2 | x): given a prompt x, the proba of completion y1>y2
- r∗(x, y1): reward function, given x, proba of y1, it's log-utilities
- σ: sigmoid function
```

2. theoretical analysis
```
- how to calculate y|x? use logits of every token. 
- logπ(y∣x) = t=1∑n ​logπ(yt​∣x,y<t​)
- no reward model needed, use LLM as reward model, logits as proba
- Actor-Critic algos are unstable (like PPO use KL-constraint), policy gradient may either have large variance or hard to optimize (contains sum up)
- KL-constrained is implicitly solved by defining reward by preference comparisons, maximize a reward signal and constrain deviation from reference model
```

3. limitation
```
- DPO has loss function to maximize the difference between chosen and rejected, it may cause both chosen and rejected proba is negative, but their difference is big.
- DPO chosen answer is usually longer than rejected answer because using sum(log(token_prob)), may cause duplicated token generation in answer.
```
