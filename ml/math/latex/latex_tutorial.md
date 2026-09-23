# LaTeX Tutorial

## $$
$$
y=2x+1
$$

## ^
$$
x^2
$$

## _
$$
a_1, \theta_t
$$

## {}
$$
a_{10} + x^{20}
$$

## \frac
*\frac{top}{bottom}*
$$
y = \frac{x^2 + 1 }{3}
$$

## \sqrt
$$
\sqrt{2}
$$

## geek letters
$$
\alpha, \beta, \theta, \epsilon, \mu, \sigma, \lambda, \delta, \nabla
$$

## calculus
$$
\Delta, \nabla, \infty, \partial, \int 
$$

$$
\int f(x) \, \mathrm{d}x
$$

$$
\int_a^b f(x) \, \mathrm{d}x
$$

$$
\mathrm{d}y = f'(x) \, \mathrm{d}x
$$

$$
\frac{\mathrm{d}y}{\mathrm{d}x}
$$

$$
\frac{\mathrm{d}^2 y}{\mathrm{d}^2 x}
$$

$$
\frac{\partial f}{\partial x}
$$

## special lettering
$$
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
$$

$$
\mathbf{x} \in \mathbb{R}^{n}
$$

## comparison and relationships
$$
\sim, \approx, \neq, \leq, \geq, \propto, \in, \notin, \subseteq, \to, \iff, M \times N
% \sim: context dependent; often "distributed as"
$$

$$
X \sim \mathcal{N}(\mu, \sigma^2) 
$$

## functions
$$
\ln(x), \exp(x), \sin \theta, \cos \theta, \max(x, y), \min(x, y)
$$

$$
\sum_{i=1}^{n} x_i % \sum_{down}^{top}
$$

$$
\prod_{i=1}^{n} x_i % \prod_{dowm}^{top}
$$

$$
\int_{0}^{1} x^2 \, dx  % \int_{down}^{top}, integral symbol
                        % \, add small horizontal space
$$

$$
\lim_{x \to 0} f(x) % \lim_{down}, limit symbol
$$

$$
\frac{\partial L}{\partial \theta}
$$

## brackets
$$
\left( ... \right),
\left[ ... \right],
\left\{ ... \right\},
\left| ... \right|,   % absolute value
\left\| ... \right\|, % norm
$$

## vectors, matrices, shapes
$$
A \in  \mathbb{R}^{m \times n}, % shape
A_{ij}, % entry
A^\top, % transpose
A^{-1}, % inverse
\mathbf{x}^\top \mathbf{y}, % dot product
A \odot B, % elementwise multml
\lVert \mathbf{x} \rVert_2, % euclidean length, length for vector
\lVert A \rVert_F, % frobenius norm, length for matrix
$$

$$
A = \begin{bmatrix}
1 & 2 \\ 3 & 4
\end{bmatrix}
% & separate cols, \\ separate rows
$$

## probability and expectations
$$
p(y \mid x), % condition probability
p(x, y), % joint probability
\mathbb{E}_{x \sim p}[f(x)], % average of f(x) when x follows p
\operatorname{Var}(X), \sigma^2(x), % variance
x_i \overset{\mathrm{iid}}{\sim} p % independent samples with the same distribution
$$

## optimization
$$
\arg\min, 
\arg\max, 
\nabla_{\theta}\mathcal{L}, % gradient
\nabla_{\theta}^2\mathcal{L}, % hessian
$$

## decorations and indexing
$$
\hat{y}, % prediction, estimate
\bar{x}, % mean
\tilde{x}, % modified or noisy quantity
\theta^\star, % optimal parameters
x^{(i)}, % sample index
h^{(\ell)}, % representation at layer
$$

## logarithms and imformation theory
$$
-\log p_{\theta}{(y \mid x)} % negative log likelihood
$$

$$
H(p) = -\sum_x p(x)\log p(x) % entropy
$$

$$
D_{\mathrm{KL}}(p \parallel q) = \sum_x p(x)\log\frac{p(x)}{q(x)}
% kl divergence from (p) to (q)
% how much distribution q differs from a reference distribution p
$$