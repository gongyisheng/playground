# Matrix

## Spectral theorem 谱定理

Every real symmetric matrix $A\in\mathbb{R}^{n\times n}$ has an orthonormal basis of eigenvectors $q_1,\ldots,q_n$ with real eigenvalues $\lambda_1,\ldots,\lambda_n$. Equivalently,

$$
\boxed{A=Q\Lambda Q^T,\qquad Q^TQ=QQ^T=I.}
$$

where $Q=\begin{bmatrix}q_1&\cdots&q_n\end{bmatrix}$ and $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$.

**Derivation.** Each eigenvector satisfies $Aq_i=\lambda_iq_i$. Combining these equations column by column gives

$$
AQ
=\begin{bmatrix}Aq_1&\cdots&Aq_n\end{bmatrix}
=\begin{bmatrix}\lambda_1q_1&\cdots&\lambda_nq_n\end{bmatrix}
=Q\Lambda.
$$

Since the eigenvectors are mutually orthogonal and have unit length,

$$
(Q^TQ)_{ij}=q_i^Tq_j=
\begin{cases}
1,&i=j,\\
0,&i\ne j.
\end{cases}
$$

Thus $Q^TQ=I$, so $Q^{-1}=Q^T$. Multiplying $AQ=Q\Lambda$ on the right by $Q^T$ yields $A=Q\Lambda Q^T$.

Geometrically, $Q^Tx$ expresses $x$ in the eigenvector basis, $\Lambda$ scales each coordinate by its eigenvalue, and $Q$ converts the result back to the original coordinates.

## Quadratic form 二次型

A quadratic form maps a vector $x\in\mathbb{R}^n$ to the scalar $q(x)=x^TAx$, where $A\in\mathbb{R}^{n\times n}$.

**Example.** For $x=(x_1,x_2)^T$ and
$A=\begin{pmatrix}3&2\\2&2\end{pmatrix}$,

$$
x^TAx=3x_1^2+4x_1x_2+2x_2^2
=x_1^2+2(x_1+x_2)^2.
$$

Every term has total degree $2$, hence the name "quadratic form" (二次型).

**Geometry (symmetric $A$, 2D).**

- **Positive definite (正定):** both eigenvalues positive → upward bowl (开口向上的碗形); positive level curves are ellipses.
- **Negative definite (负定):** both eigenvalues negative → downward bowl (开口向下的碗形); negative level curves are ellipses.
- **Indefinite (不定):** eigenvalues have opposite signs → saddle (马鞍面); nonzero level curves are hyperbolas.
- **Singular semidefinite (奇异半定):** one eigenvalue zero → trough or ridge (槽形或脊形) with a flat direction; both zero → flat plane.

Eigenvectors give the principal directions; larger eigenvalue magnitudes mean steeper curvature.

## Positive definite 正定

assume $A\in\mathbb{R}^{n\times n}$ is symmetric: $A=A^T$.

$A$ is **positive definite**, written $A\succ 0$, if its quadratic form(二次型) is strictly positive for every nonzero vector:

$$
A\succ 0\quad\Longleftrightarrow\quad x^TAx>0\quad\text{for all }x\in\mathbb{R}^n\setminus\{0\}
$$

This extends scalar positivity:

$$
a>0\quad\Longleftrightarrow\quad ax^2>0\quad\text{for all }x\in\mathbb{R}\setminus\{0\}
$$

### Theorem 1: Eigenvalue criterion

A real symmetric matrix is positive definite if and only if all its eigenvalues are positive:

$$
\boxed{A\succ 0\quad\Longleftrightarrow\quad\lambda_i>0\quad\text{for all }i.}
$$

**Derivation.** By the spectral theorem,

$$
A=Q\Lambda Q^T,\qquad Q^TQ=I,\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n).
$$

Set $z=Q^Tx$. Since $Q$ is invertible, $x\ne 0$ if and only if $z\ne 0$, and

$$
x^TAx=x^TQ\Lambda Q^Tx=z^T\Lambda z
=\sum_{i=1}^n\lambda_i z_i^2.
$$

If every $\lambda_i>0$, this sum is positive for every nonzero $z$, so $A\succ 0$.

Conversely, if $A\succ 0$, taking $x=q_i$, the unit eigenvector for $\lambda_i$, gives

$$
0<q_i^TAq_i=\lambda_i q_i^Tq_i=\lambda_i.
$$

### Theorem 2: Sylvester's criterion

A real symmetric matrix is positive definite if and only if all its leading principal minors are positive:

$$
\boxed{A\succ 0\quad\Longleftrightarrow\quad
\det(A_k)>0,\quad k=1,\ldots,n.}
$$

Here $A_k$ is the upper-left $k\times k$ submatrix of $A$.

**Derivation.** for the $2\times 2$ case, Let

$$
A=\begin{pmatrix}a&b\\b&c\end{pmatrix}.
$$

Positive definiteness requires $a>0$, by taking $x=(1,0)^T$. For $a>0$, completing the square gives

$$
x^TAx=ax_1^2+2bx_1x_2+cx_2^2
=a\left(x_1+\frac{b}{a}x_2\right)^2
+\frac{ac-b^2}{a}x_2^2.
$$

If $ac-b^2>0$, the expression is positive for every nonzero $x$. Conversely, choosing $x=(-b/a,1)^T$ shows that positive definiteness requires $ac-b^2>0$. Thus,

$$
\boxed{A\succ 0\quad\Longleftrightarrow\quad a>0\quad\text{and}\quad ac-b^2>0.}
$$

These are exactly the conditions $\det(A_1)>0$ and $\det(A_2)>0$.

## First moment 一阶矩

For a random vector $z\in\mathbb{R}^n$ with finite first moments, the first moment is its mean vector:

$$
\mu=\mathbb{E}[z],\qquad
\widehat\mu=\frac1N\sum_{i=1}^N z_i.
$$

- **Mean:** $\mu$ is the true average; $\widehat\mu$ estimates it from samples.
- **Linearity:** for fixed $B$ and $b$, $\mathbb{E}[Bz+b]=B\mu+b$.
- **Centering:** subtracting $\mu$ gives zero mean, but values can still vary.

## Second moment 二阶矩

For a random vector $z\in\mathbb{R}^n$ with finite second moments, define

$$
A=\mathbb{E}[zz^T],\qquad
\widehat A=\frac1N\sum_{i=1}^N z_i z_i^T=\frac1NZZ^T,
$$

where $Z=[z_1,\ldots,z_N]$ contains the observed samples.

- **Symmetry and positivity:** $A=A^T$ and $v^TAv=\mathbb{E}[(v^Tz)^2]\ge0$.
- **Covariance:** $A=\operatorname{Cov}(z)+\mu\mu^T$, so they are equal when $\mu=0$.
- **Directions:** each eigenvalue measures the mean squared projection along its unit eigenvector. $A=cI$ means equal strength in all directions.
- **Invertibility:** $\widehat A$ is invertible exactly when the samples span $\mathbb{R}^n$.
- **Preconditioning:** for $A\succ0$, $A^{-1}$ scales each eigenvector direction by $1/\lambda_i$. Using $(A+\gamma I)^{-1}$ with $\gamma>0$ avoids dividing by tiny or zero eigenvalues.
