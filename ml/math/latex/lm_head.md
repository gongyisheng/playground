# LM head
ref: https://arxiv.org/pdf/2603.10145

assume we have hidden state $h$ which will be passed to lm_head, where $x$ is the input token, $\theta$ is the weight of network, D is the hidden state dim.

```math
h=F_{\theta}(x), h \in \mathbb R^{D}
```

then pass it to lm head, which is a linear layer. The shape of W is $\mathbb{R}^{V\times D}$

```math
z = Wh, z \in \mathbb R^{V}
```

each row of $W$ is a token, so the score of token i is:

```math
z_{i} = W_{i}h
```

then softmax turn score to proba:

```math
p_{i} = \frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_{j}}}
```

assume the correct token id is $t$, thus the cross entropy loss is 

```math
L=-\log p_t
```

now move to backward, we first compute the gradient of the loss with respect to the logits

```math
L=-\log \frac{e^{z_t}}{\sum_{j=1}^V e^{z_j}} = -z_t + \log \sum_{j=1}^V e^{z_j}
```

differentiate with respect to each $z_i$, we can get the grad of each token:

```math
\frac{\partial L}{\partial z_i} = p_i-1[i=t] = \begin{cases}
p_i-1, & i=t\quad\text{(correct token)}\\
p_i, & i\ne t\quad\text{(wrong token)}
\end{cases}
```

since we already know $z_i=\sum_j W_{ij}h_j$, according to the chained rule,

```math
\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial z_{i}} \frac{\partial z_i}{\partial W_{ij}} = (g_z)_i h_j
```

write as a matrix form:

```math
\nabla_W L = g_z h^T
```

where as its shape is $(V\times 1)(1\times D)=V\times D$

if we make $h$ fixed, with the help of optimizer, the projection of correct token weight in lm_head $w_t$ onto the direction of $h$ increases, thereby increasing the dot product of $w_th$, which is its score  

then let's look at the gradient passed to transformer

```math
\frac{\partial z_i}{\partial h_j} = W_{ij}
```

since every logit depends on $h_j$, need to sum up all paths

```math
\frac{\partial L}{\partial h_j} = \sum_i \frac{\partial L}{\partial z_i} \frac{\partial z_i}{\partial h_j} = \sum_i (g_z)_iW_{ij}
```

write as a matrix form:

```math
g_h = W^T g_z
```

differentiate to transformer model weight $\theta$:

```math
\nabla_{\theta}L=(\frac{\partial h}{\partial \theta})^{T} g_h
```

now lets talk about how lm_head becomes a gradient bottleneck, the gradient passed to lm_head is 

```math
\nabla_W L=(p-y)h^\top
```

whereas $p$ is each token's proba and y is the ground truth  

the gradient passed to hidden state and the rest of the network is

```math
g_h=W^\top(p-y)
```

whereas $W^\top \in \mathbb{R}^{D\times V}$, and $(p-y) \in \mathbb{R}^{V\times 1}$, so here back propagation maps V-dim signal into D-dim feedback.

when $D \lt V$, the linear map cannot preserve all directions. there must exist a nonzero vector $q \in \mathbb R^{V}$, let $W^{T} q=0$, which means two different grad diffs by $q$ will get same result:

```math
W^{T}(g+q)=W^{T} g
```

thus, the gradient that transformer model can get is a low rank expression of the full logit's gradient, it is limited by d_model. need to sweep d_model to see how much loss it causes.