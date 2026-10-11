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
