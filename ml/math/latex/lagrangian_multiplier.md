# Lagrange Multiplier
Suppose we want to minimize a function $f(x,y)$ subject to a constraint $g(x,y)=0$.

The main idea: at a constrained optimum, the gradient of the objective is parallel to the gradient of the constraint:

$$
\nabla f(x,y) = -\lambda \nabla g(x,y)
$$

The number $\lambda$ is called a Lagrange multiplier.

To understand this: for the constraint $g(x,y)=0$, the gradient $\nabla g$ is perpendicular to the constraint curve. At a local optimum, moving along the curve cannot change $f$ to first order, so $\nabla f$ is perpendicular to the curve too. Therefore, the two gradients are parallel. This condition identifies possible optima; it does not prove that a point is a minimum or maximum.

# Lagrangian Function
Combine the objective and constraint into one function:

$$
\mathcal{L}(x, y, \lambda)=f(x,y)+\lambda g(x,y)
$$

To find candidates for an optimum, solve:

$$
\frac{\partial\mathcal{L}}{\partial x}=0, \quad
\frac{\partial\mathcal{L}}{\partial y}=0, \quad
\frac{\partial\mathcal{L}}{\partial \lambda}=0.
$$

## Example
Minimize $f(x,y)=x^2+y^2$ subject to $x+y=10$.

Lagrangian function:
$$
\mathcal{L}(x,y,\lambda)=x^2+y^2+\lambda(x+y-10).
$$

The stationary equations (驻点方程) are 
$$
2x+\lambda=0,\\
2y+\lambda=0,\\
x+y-10=0
$$

They give $x=y=5$. Since $x+y=10$ implies $x^2+y^2=50+\frac{1}{2}(x-y)^2\ge 50$, this point is a minimum, with value $50$.
