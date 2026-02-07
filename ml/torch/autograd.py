"""
Demonstration of PyTorch autograd system and gradient computation.

grad_fn is an attribute that stores a reference to the function that created
the tensor through an operation. It's the backbone of automatic differentiation.
"""

import torch


def test_detach():
    # detach() returns a new tensor that shares the same storage
    # but is detached from the computation graph (no gradient tracking)
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    print("y requires_grad:", y.requires_grad)  # True

    z = y.detach()
    print("z requires_grad:", z.requires_grad)  # False
    print("z shares data with y:", z.data_ptr() == y.data_ptr())  # True

    # common use: stop gradient flow in part of a network
    # e.g. target networks in RL, frozen encoders, etc.
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = a * 3
    loss = (b - b.detach()).sum()  # gradient won't flow through detached b
    loss.backward()
    print("a.grad:", a.grad)  # [3., 3.] — only from the non-detached b


def test_grad_property():
    """Demonstrate grad_fn, is_leaf, and detach properties."""
    print("=" * 50)
    print("Gradient Properties: leaf, grad_fn, detach")
    print("=" * 50)

    # Leaf tensor - created directly, no grad_fn
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    print(f"\nLeaf tensor (user-created):")
    print(f"  x = {x}")
    print(f"  x.grad_fn = {x.grad_fn}")  # None
    print(f"  x.is_leaf = {x.is_leaf}")  # True

    # Non-leaf tensor - created from operation
    y = x * 2
    print(f"\nNon-leaf tensor (from operation):")
    print(f"  y = x * 2 = {y}")
    print(f"  y.grad_fn = {y.grad_fn}")  # MulBackward0
    print(f"  y.is_leaf = {y.is_leaf}")  # False

    # Tensor without requires_grad has no grad_fn
    a = torch.tensor([1.0, 2.0])
    b = a * 2
    print(f"\nTensor without requires_grad:")
    print(f"  a.requires_grad = {a.requires_grad}")
    print(f"  b.grad_fn = {b.grad_fn}")  # None

    # Detach removes tensor from computation graph
    z = y.detach()
    print(f"\nDetached tensor:")
    print(f"  z = y.detach()")
    print(f"  z.grad_fn = {z.grad_fn}")  # None
    print(f"  z.requires_grad = {z.requires_grad}")  # False
    print(f"  shares memory: {y.data_ptr() == z.data_ptr()}")  # True

    # clone().detach() for independent copy
    w = y.clone().detach()
    print(f"\nClone then detach (independent copy):")
    print(f"  w = y.clone().detach()")
    print(f"  shares memory: {y.data_ptr() == w.data_ptr()}")  # False


def test_grad_add_sum():
    """Demonstrate gradient for add and sum operations.

    Add and sum have local derivative of 1, so they pass gradients through unchanged.
    - add: ∂(x+c)/∂x = 1
    - sum: ∂sum(x)/∂xᵢ = 1 (broadcasts scalar gradient back to vector)
    """
    print("=" * 50)
    print("Gradient: Add and Sum (derivative = 1)")
    print("=" * 50)

    # Add with constant
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x + 5
    z = y.sum()
    z.backward()
    print(f"\nAdd with constant (y = x + 5):")
    print(f"  x = {x.tolist()}")
    print(f"  y = x + 5 = {y.tolist()}")
    print(f"  z = y.sum() = {z.item()}")
    print(f"  x.grad = {x.grad.tolist()}")  # [1, 1]
    print(f"  (gradient passes through unchanged)")

    # Add two tensors
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor([4.0, 5.0], requires_grad=True)
    c = (a + b).sum()
    c.backward()
    print(f"\nAdd two tensors (c = (a + b).sum()):")
    print(f"  a = {a.tolist()}, b = {b.tolist()}")
    print(f"  c = {c.item()}")
    print(f"  a.grad = {a.grad.tolist()}")  # [1, 1]
    print(f"  b.grad = {b.grad.tolist()}")  # [1, 1]
    print(f"  (both get gradient of 1)")

    # Sum broadcasts gradient
    x2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y2 = x2.sum()  # scalar
    y2.backward()
    print(f"\nSum broadcasts gradient:")
    print(f"  x = {x2.tolist()}")
    print(f"  y = x.sum() = {y2.item()}")
    print(f"  x.grad = {x2.grad.tolist()}")  # [1, 1, 1]


def test_grad_scale_divide():
    """Demonstrate gradient for scale and divide operations.

    - scale: ∂(c*x)/∂x = c
    - divide: ∂(x/c)/∂x = 1/c
    """
    print("=" * 50)
    print("Gradient: Scale and Divide")
    print("=" * 50)

    # Scale by constant
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x * 3
    z = y.sum()
    z.backward()
    print(f"\nScale by constant (y = x * 3):")
    print(f"  x = {x.tolist()}")
    print(f"  y = x * 3 = {y.tolist()}")
    print(f"  z = y.sum() = {z.item()}")
    print(f"  x.grad = {x.grad.tolist()}")  # [3, 3]
    print(f"  (gradient scaled by 3)")

    # Divide by constant
    a = torch.tensor([6.0, 9.0], requires_grad=True)
    b = a / 3
    c = b.sum()
    c.backward()
    print(f"\nDivide by constant (b = a / 3):")
    print(f"  a = {a.tolist()}")
    print(f"  b = a / 3 = {b.tolist()}")
    print(f"  c = b.sum() = {c.item()}")
    print(f"  a.grad = {a.grad.tolist()}")  # [0.333, 0.333]
    print(f"  (gradient scaled by 1/3)")

    # Chain of scales
    x2 = torch.tensor([1.0, 2.0], requires_grad=True)
    y2 = x2 * 2 * 3  # scale by 6
    z2 = y2.sum()
    z2.backward()
    print(f"\nChain of scales (y = x * 2 * 3):")
    print(f"  x = {x2.tolist()}")
    print(f"  y = x * 2 * 3 = {y2.tolist()}")
    print(f"  x.grad = {x2.grad.tolist()}")  # [6, 6]
    print(f"  (gradients multiply: 2 * 3 = 6)")


def test_grad_power():
    """Demonstrate gradient for power operations.

    - power: ∂(x^n)/∂x = n * x^(n-1)
    - sqrt: ∂(√x)/∂x = 1/(2√x)
    """
    print("=" * 50)
    print("Gradient: Power Operations")
    print("=" * 50)

    # Square: x^2
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x ** 2
    z = y.sum()
    z.backward()
    print(f"\nSquare (y = x^2):")
    print(f"  x = {x.tolist()}")
    print(f"  y = x^2 = {y.tolist()}")
    print(f"  x.grad = {x.grad.tolist()}")  # [4, 6] = 2*x
    print(f"  (∂x²/∂x = 2x)")

    # Cube: x^3
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = a ** 3
    c = b.sum()
    c.backward()
    print(f"\nCube (b = a^3):")
    print(f"  a = {a.tolist()}")
    print(f"  b = a^3 = {b.tolist()}")
    print(f"  a.grad = {a.grad.tolist()}")  # [12, 27] = 3*a^2
    print(f"  (∂x³/∂x = 3x²)")

    # Square root: x^0.5
    x2 = torch.tensor([4.0, 9.0], requires_grad=True)
    y2 = torch.sqrt(x2)
    z2 = y2.sum()
    z2.backward()
    print(f"\nSquare root (y = √x):")
    print(f"  x = {x2.tolist()}")
    print(f"  y = √x = {y2.tolist()}")
    print(f"  x.grad = {x2.grad.tolist()}")  # [0.25, 0.166] = 1/(2√x)
    print(f"  (∂√x/∂x = 1/(2√x))")


def test_grad_two_tensor_multiply():
    """Demonstrate gradient when two tensors multiply.

    For z = x * y (element-wise):
    - ∂z/∂x = y (gradient is the other tensor)
    - ∂z/∂y = x (gradient is the other tensor)
    """
    print("=" * 50)
    print("Gradient: Two-Tensor Multiplication (swapped)")
    print("=" * 50)

    # Element-wise multiplication
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0], requires_grad=True)
    z = (x * y).sum()  # z = 2*4 + 3*5 = 23
    z.backward()
    print(f"\nElement-wise multiply (z = (x * y).sum()):")
    print(f"  x = {x.tolist()}")
    print(f"  y = {y.tolist()}")
    print(f"  z = {z.item()}")
    print(f"  x.grad = {x.grad.tolist()} (= y values)")
    print(f"  y.grad = {y.grad.tolist()} (= x values)")

    # Division by tensor: x / y
    a = torch.tensor([6.0, 10.0], requires_grad=True)
    b = torch.tensor([2.0, 5.0], requires_grad=True)
    c = (a / b).sum()  # c = 6/2 + 10/5 = 5
    c.backward()
    print(f"\nDivision (c = (a / b).sum()):")
    print(f"  a = {a.tolist()}")
    print(f"  b = {b.tolist()}")
    print(f"  c = {c.item()}")
    print(f"  a.grad = {a.grad.tolist()}")  # [0.5, 0.2] = 1/b
    print(f"  b.grad = {b.grad.tolist()}")  # [-1.5, -0.4] = -a/b²
    print(f"  (∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²)")

    # Matmul example
    x2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y2 = torch.tensor([[5.0], [6.0]], requires_grad=True)
    z2 = (x2 @ y2).sum()  # matmul then sum
    z2.backward()
    print(f"\nMatrix multiply (z = (x @ y).sum()):")
    print(f"  x = {x2.tolist()}")
    print(f"  y = {y2.tolist()}")
    print(f"  x @ y = {(x2 @ y2).tolist()}")
    print(f"  x.grad = {x2.grad.tolist()}")
    print(f"  y.grad = {y2.grad.tolist()}")


if __name__ == "__main__":
    test_detach()
    # test_grad_property()
    # test_grad_add_sum()
    # test_grad_scale_divide()
    # test_grad_power()
    # test_grad_two_tensor_multiply()
