# Muon Optimizer

## algorithm

1. calculate gradient and momentum
$$
G_t = \nabla_{W} L(W_t) \\
M_t = \beta M_{t-1} + (1-\beta)G_t \\
D_t = \beta M_{t} + (1-\beta)G_t \\ % when use nesterov
$$

2. orthogonalize
$$
D_t = U\Sigma V^T \\
O_t = UV^T \\ % preserve directions, set singular values to 1, assuming full rank
$$

3. update weights
$$
W_{t+1} = (1-\eta\lambda)W_t - \eta s O_{t} \\
s = \sqrt{max(1, \frac{m}{n})} \\
% \eta: learning rate
% \lambda: weight decay
% m: input dim, n: output dim
$$

## discussions
- nesterov: observed with nesterov outperforms without nesterov, it helps momuntum adapt to the latest update