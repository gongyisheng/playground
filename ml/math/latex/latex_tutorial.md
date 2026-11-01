# LaTeX Tutorial

## Display math
```math
y=2x+1
```

## ^
```math
\begin{aligned}
x^2
\end{aligned}
```

## _
```math
a_1, \theta_t
```

## {}
```math
a_{10} + x^{20}
```

## \frac
*\frac{top}{bottom}*
```math
y = \frac{x^2 + 1 }{3}
```

## \sqrt
```math
\sqrt{2}
```

## dots
```math
\ldots, \cdots, \vdots, \ddots
```

## geek letters
```math
\alpha, \beta, \theta, \epsilon, \mu, \sigma, \lambda, \delta, \nabla, \eta
```

## calculus
```math
\Delta, \nabla, \infty, \partial, \int 
```

```math
\int f(x) \, \mathrm{d}x
```

```math
\int_a^b f(x) \, \mathrm{d}x
```

```math
\mathrm{d}y = f'(x) \, \mathrm{d}x
```

```math
\begin{aligned}
\frac{\mathrm{d}y}{\mathrm{d}x}
\end{aligned}
```

```math
\begin{aligned}
\frac{\mathrm{d}^2 y}{\mathrm{d}^2 x}
\end{aligned}
```

```math
\begin{aligned}
\frac{\partial f}{\partial x}
\end{aligned}
```

## special lettering
```math
\mathbb{R}, % real numbers
\mathbb{N}, % natural numbers
\mathbb{Z}, % integers
\mathbb{C}, % complex numbers
\mathbb{E}, % expected value
\mathbf{x}, % bold vector
\mathcal{L}, % loss function
\mathcal{N},
\mathrm{kg},

% mathbb: blackboard bold, number sets
% mathcal: calligraphic (curly), collections
% mathbf: bold, vectors and matrics
% mathrm: upright, units, descriptive labels, differentials
```

```math
\begin{aligned}
\mathbf{x} \in \mathbb{R}^{n}
\end{aligned}
```

## comparison and relationships
```math
\sim, \approx, \neq, \leq, \geq, \propto, \in, \notin, \subseteq, \to, \iff, M \times N
% \sim: context dependent; often "distributed as"
```

```math
X \sim \mathcal{N}(\mu, \sigma^2) 
```

## functions
```math
\ln(x), \exp(x), \sin \theta, \cos \theta, \max(x, y), \min(x, y)
```

```math
\sum_{i=1}^{n} x_i % \sum_{down}^{top}
```

```math
\prod_{i=1}^{n} x_i % \prod_{dowm}^{top}
```

```math
\int_{0}^{1} x^2 \, dx  % \int_{down}^{top}, integral symbol
                        % \, add small horizontal space
```

```math
\lim_{x \to 0} f(x) % \lim_{down}, limit symbol
```

```math
\begin{aligned}
\frac{\partial L}{\partial \theta}
\end{aligned}
```

## brackets
```math
\left( ... \right),
\left[ ... \right],
\left\{ ... \right\},
\left| ... \right|,   % absolute value
\left\| ... \right\|, % norm
```

## vectors, matrices, shapes
```math
A \in  \mathbb{R}^{m \times n}, % shape
A_{ij}, % entry
A^\top, % transpose
A^{-1}, % inverse
\mathbf{x}^\top \mathbf{y}, % dot product
A \odot B, % elementwise multml
\lVert \mathbf{x} \rVert_2, % euclidean length, length for vector
\lVert A \rVert_F, % frobenius norm, length for matrix
```

```math
A = \begin{bmatrix}
1 & 2 \\ 3 & 4
\end{bmatrix}
% & separate cols, \\ separate rows
```

## probability and expectations
```math
p(y \mid x), % condition probability
p(x, y), % joint probability
\mathbb{E}_{x \sim p}[f(x)], % average of f(x) when x follows p
\mathrm{Var}(X), \sigma^2(x), % variance
x_i \overset{\mathrm{iid}}{\sim} p % independent samples with the same distribution
```

## optimization
```math
\arg\min, 
\arg\max, 
\nabla_{\theta}\mathcal{L}, % gradient
\nabla_{\theta}^2\mathcal{L}, % hessian
```

## decorations and indexing
```math
\hat{y}, % prediction, estimate
\bar{x}, % mean
\tilde{x}, % modified or noisy quantity
\theta^\star, % optimal parameters
x^{(i)}, % sample index
h^{(\ell)}, % representation at layer
```

## logarithms and imformation theory
```math
-\log p_{\theta}{(y \mid x)} % negative log likelihood
```

```math
H(p) = -\sum_x p(x)\log p(x) % entropy
```

```math
D_{\mathrm{KL}}(p \parallel q) = \sum_x p(x)\log\frac{p(x)}{q(x)}
% kl divergence from (p) to (q)
% how much distribution q differs from a reference distribution p
```