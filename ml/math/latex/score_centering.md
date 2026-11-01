# Score Centering
ref: https://arxiv.org/pdf/2609.20807

```math
p_\theta(v\mid h) \\
q(v\mid h) \\
s_v=\nabla_\theta\log p_\theta(v\mid h)
```

if on-policy:
```math
\mathbb E_p[s]=\sum_vp_v\nabla_\theta\log p_v=\nabla_\theta\sum_vp_v = 0
```