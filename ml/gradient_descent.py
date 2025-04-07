# minimize function: f(x) = x^2
# should converge to 0

import numpy as np

def minimize(iter, learning_rate, start_point):
    x = start_point
    for i in range(iter):
        gradient = 2 * x  # derivative of f(x) = x^2
        x = x - learning_rate * gradient
        print(f"Iteration {i+1}: x = {x}, f(x) = {x**2}")
    return x

if __name__ == "__main__":
    minimize(100, 0.1, 10)