# Reinforcement Learning
focus on ppo/grpo here

objective: given a model $\pi_\theta$ and a prompt x, calculate the expectation of its answers' (rollouts) reward
$$
J(\theta)=\mathbb E_{y\sim\pi_\theta(\cdot|x)}[R(x,y)]
$$

equivalent to 
$$
J(\theta)=\sum_y \pi_\theta(y\mid x) R(x,y)
$$

differentiate with respect to $\theta$
$$
\nabla_\theta J(\theta)=\sum_y R(x,y)\nabla_\theta\pi_\theta(y\mid x)
$$

since we have log-derivative trick:
$$
\nabla_\theta \pi_\theta(y|x)=\pi_\theta(y|x)\nabla_\theta\log\pi_\theta(y|x)
$$

thus we have the gradient
$$
\nabla_\theta J(\theta)=\mathbb E_{y\sim\pi_\theta}\left[R(x,y)\nabla_\theta\log\pi_\theta(y|x)\right]
$$

for llm, log proba of a sequence is sum of log proba of its tokens
$$
\log\pi_\theta(y|x)=\sum_{t=1}^{T}\log\pi_\theta(y_t|x,y_{<t})
$$

thus, 
$$
\nabla_\theta J=\mathbb E\left[\sum_{t=1}^{T}A_t\nabla_\theta\log\pi_\theta(y_t|s_t)\right]
$$

where $A_t$ is the advantage, used to replace the raw reward and reduce the variance of gradient estimation
