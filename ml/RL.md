# Reinforcement Learning
## PPO
Goal: take the biggest possible improvement step on a policy using the data we currently have, without stepping so far  
Pros: use first-order methods, compute friendly  
Variants:  
- PPO-Penalty: penalize by KL-divergence, the farther it goes, the more penality it gets  
- PPO-Clip: penalize by setting clippings (max, min), discourage gradients from moving too far  
We mainly use PPO-Clip because it's stable, KL-constrain can cause big updates.   

(talk about PPO-Clip below)  

Objective function: L(θ)=E[min(r(θ)A^,clip(r(θ),1−ϵ,1+ϵ)A^)]
r(θ): ratio (how much the new policy changes the probability of that action), πθ​(a∣s)/πold(a∣s), r>1 r<1   
A^: advantage (how much better or worse the action was compared to expectation), A^>0 A^<0  
r(θ)A^: r(θ) to get how much to change, A^ to get what direction to change
Use GAE to estimate advantage

## DPO
## GRPO