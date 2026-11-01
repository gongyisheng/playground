# Taylor Expansion: From Scalars to Matrices

assume a smooth function $f$, let $h$ be a small change from a point $a$, the taylor expansion is

```math
f(a+h)=f(a)+f'(a)h+\frac12f''(a)h^2+O(h^3)
```
where $f'(a)h$ is first order change, $\frac12f''(a)h$ is second order change. 

a generalized version is 

```math
\begin{aligned}
T_n(a+h)=\sum_{k=0}^{n}\frac{f^{(k)}(a)}{k!}h^k.
\end{aligned}
```

then we generalize it to vector inputs:

```math
f(x+h)=f(x)+\nabla f(x)^Th+\frac12h^TH_f(x)h+O(\|h\|^3)
```

**Derivation**

write updates as h=sv, where v is a direction and s is a small scalar step, define

```math
g(t)=f(x+tv)
```

because g takes a scalar input, ordinary taylore expansion applies (expand at $t \approx 0$):
```math
f(x+sv)=g(s) \approx g(0)+g'(0)s+\frac12g''(0)s^2
```

first derivative: 
```math
g'(t)=\sum_i{\frac{\partial f}{\partial x_i}}(x+tv)v_i=\nabla f(x+tv)^{T}v
```

second derivative: (h=sv)
```math
g''(t)=\sum_i{v_i\frac{d}{dt}[\frac{\partial f}{\partial x_i}(x+tv)]}=\sum_i \sum_j \frac{\partial^2f}{\partial x_i \partial x_j}(x+tv)v_j = \frac12h^{T}H_{f}h
```
