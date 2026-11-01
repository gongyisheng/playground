# weight quantization analysis
consider linear transform
```math
\begin{aligned}
Y=WX
\end{aligned}
```

## forward  
```math
\begin{aligned}
Y &= WX  \\
W &\in \mathbb{R}^{m \times k}, \\
X &\in \mathbb{R}^{k \times n}
\end{aligned}
% k: input dim
% m: output dim 
```

## backward
backward:
```math
G=\frac{\partial L}{\partial Y} \quad 
G_{ij}=\frac{\partial L}{\partial Y_{ij}} \quad 
G\in\mathbb{R}^{m\times n}
```

since  
```math
Y_{ij}=\sum_{r=1}^k W_{ir}X_{rj},
```

thus

```math
\frac{\partial Y_{ij}}{\partial W_{ab}}=\begin{cases} X_{bj}, & i=a,\\0, & i\ne a.\end{cases}
```

```math
\frac{\partial Y_{ij}}{\partial X_{ab}}=\begin{cases} W_{ia}, & j=b,\\0, & j\ne b.\end{cases}
```

Again applying the chain rule
```math
\begin{aligned}
\frac{\partial L}{\partial X_{ab}}&=\sum_{i=1}^m\sum_{j=1}^n\frac{\partial L}{\partial Y_{ij}}\frac{\partial Y_{ij}}{\partial X_{ab}} \\
&= \sum_{i=1}^mW_{ia}G_{ib} \\
&= (W^\top G)_{ab}
\end{aligned}
```

```math
\begin{aligned}
\frac{\partial L}{\partial W_{ab}}&=\sum_{i=1}^m\sum_{j=1}^n\frac{\partial L}{\partial Y_{ij}}\frac{\partial Y_{ij}}{\partial W_{ab}} \\
&= \sum_{j=1}^nG_{aj}X_{bj} \\
&= (GX^\top)_{ab}
\end{aligned}
```

therefore,
```math
\begin{aligned}
\frac{\partial L}{\partial X}=W^\top G \qquad (k\times m)(m\times n)=k\times n.
\end{aligned}
```

```math
\begin{aligned}
\frac{\partial L}{\partial W}= GX^\top \qquad (m\times n)(n\times k)=(m\times k)
\end{aligned}
```

## quantization error
assume quantization cause error E:
```math
\begin{aligned}
\widehat Y&=\widehat WX=(W+E)X \\
\Delta Y&=\widehat Y-Y=EX
\end{aligned}
```
if we use 2d blockwise quantization:
```math
\boxed{E_{\mathrm{bwd}}=E_{\mathrm{fwd}}^\top}
```
if we use 1d blockwise quantization:
```math
\boxed{E_{\mathrm{bwd}}\neq E_{\mathrm{fwd}}^\top}
```

forward:
```math
Y=(W+E_{\mathrm{fwd}})X
```

backward:
```math
\begin{aligned}
\frac{\partial L}{\partial X}=(W^\top+E_{\mathrm{bwd}}) G
\end{aligned}
```

for gradient
```math
G=\left.\frac{\partial L}{\partial Y}\right|_{Y=WX},\quad \widehat G=\left.\frac{\partial L}{\partial Y}\right|_{Y=(W+E_\mathrm{fwd})X}
```

define
```math
\begin{aligned}
G&=L'(Y), \quad \widehat G=L'(Y+\Delta Y), \\
\Delta G&=\widehat G-G
\end{aligned}
```

## error in weight update
define
```math
\begin{aligned}
D_W&=GX^\top, \quad \widehat D_W=\widehat GX^\top \\
\Delta D_W&=\Delta GX^\top
\end{aligned}
```

with learning rate $\eta_W$, the two SGD updates are 
```math
\begin{aligned}
W_{\mathrm{next}}&=W-\eta_W D_W,
\qquad
\widehat W_{\mathrm{next}}=W-\eta_W\widehat D_W. \\
\widehat W_{\mathrm{next}}-W_{\mathrm{next}}&
=-\eta_W\Delta G\,X^\top.
\end{aligned}
```

## error in the input update
define
```math
\begin{aligned}
D_X&=W^\top G, \quad \widehat D_X=(W^\top+E_{\mathrm{bwd}})\widehat G \\
\Delta D_X&=W^\top\Delta G + E_{\mathrm{bwd}}G+E_{\mathrm{bwd}}\Delta G
\end{aligned}
```