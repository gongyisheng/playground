# norm
properties of norm (rmsnorm, layernorm)

# theory 1
for normalized layers, their gradients are orthogonal to their weights, $\langle g_t, x_t \rangle = 0$  

proof: 

assume we have normalized layers $Y=WX$, and we can apply a multiplier c to the layer $Y=cWX$. since it's normalized we have 
$$
L(cW)=L(W)
$$

derivate on c:
$$
0 = \left.\frac{\partial L(cW)}{dc}\right|_{c=1}
$$

since
$$
\frac{\partial L(cW)}{dc} = \sum_i^n\frac{\partial L(cW)}{\partial W_i}\frac{d (cW_i)}{dc}
$$

thus
$$
\frac{\partial L(cW)}{dc} = \langle \nabla L(cW), W\rangle = 0
$$

set c=1:
$$
\boxed{\langle \nabla L(W), W\rangle = 0}
$$