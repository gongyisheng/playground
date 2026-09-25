# weight quantization 1d vs 2d

## forward  
$$
Y = WX  \\
W \in \mathbb{R}^{m \times k},
X \in \mathbb{R}^{k \times n}
% k: input dim
% m: output dim 
$$

## backward
backward:
$$
G=\frac{\partial L}{\partial Y} \quad 
G_{ij}=\frac{\partial L}{\partial Y_{ij}} \quad 
G\in\mathbb{R}^{m\times n}
$$

since  
$$
Y_{ij}=\sum_{r=1}^k W_{ir}X_{rj},
$$

thus

$$
\frac{\partial Y_{ij}}{\partial W_{ab}}=\begin{cases} X_{bj}, & i=a,\\0, & i\ne a.\end{cases}
$$

$$
\frac{\partial Y_{ij}}{\partial X_{ab}}=\begin{cases} W_{ia}, & j=b,\\0, & j\ne b.\end{cases}
$$

Again applying the chain rule
$$
\frac{\partial L}{\partial X_{ab}}=\sum_{i=1}^m\sum_{j=1}^n\frac{\partial L}{\partial Y_{ij}}\frac{\partial Y_{ij}}{\partial X_{ab}} \\
= \sum_{i=1}^mW_{ia}G_{ib} \\ 
= (W^\top G)_{ab}
$$

$$
\frac{\partial L}{\partial W_{ab}}=\sum_{i=1}^m\sum_{j=1}^n\frac{\partial L}{\partial Y_{ij}}\frac{\partial Y_{ij}}{\partial W_{ab}} \\
= \sum_{j=1}^nG_{aj}X_{bj} \\
= (GX^\top)_{ab}
$$

therefore,
$$
\frac{\partial L}{\partial X}=W^\top G \qquad (k\times m)(m\times n)=k\times n.
$$

$$
\frac{\partial L}{\partial W}= GX^\top \qquad (m\times n)(n\times k)=(m\times k)
$$

## blockwise1d
