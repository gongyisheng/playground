# Newton's Method

Newton's method uses the gradient (first order change) and curvature (second order change) to build a local quadratic approximation (局部二次近似). When the Hessian is positive definite, it moves to the minimum of that approximation.

Assume we have a function $f(x)$ and want to know its location of the minimum $x^\star$, starts at a initial guess $w$,

For a scalar $w$:

$$
w_{\text{new}} = w - \frac{f'(w)}{f''(w)}.
$$

Here, $f'(w)$ is the gradient and $f''(w)$ is the curvature. Dividing by curvature corrects the scale of the gradient.

**Example:** For $f(w)=w^2$, starting at $w=3$:

$$
w_{\text{new}} = 3 - \frac{6}{2} = 0.
$$

For a vector $w$, the second derivative becomes the Hessian matrix $H=\nabla^2 f(w)$:

$$
w_{\text{new}} = w - H^{-1}\nabla f(w).
$$

For the quadratic function (二次函数) $f(w)=\frac12(w-w^\star)^\top A(w-w^\star)$, with $A$ symmetric positive definite:

$$
g=A(w-w^\star), \qquad A^{-1}g=w-w^\star.
$$

Thus, one Newton step reaches $w^\star$. For general functions, the quadratic model is only a local approximation, so repeated steps and sometimes damping are needed. The formulas above require an invertible Hessian (a nonzero second derivative in one dimension).
