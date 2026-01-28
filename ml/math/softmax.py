import numpy as np

def softmax_unstable(x):
    """Unstable softmax - can overflow for large values."""
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def softmax_stable(x):
    """Numerically stable softmax.

    Subtracts the max value to prevent overflow:
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    This is mathematically equivalent because:
    exp(x - c) / sum(exp(x - c)) = exp(x)/exp(c) / (sum(exp(x))/exp(c))
                                 = exp(x) / sum(exp(x))

    Passes: 3 (max, sum, divide)
    """
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)


def softmax_online(x):
    """Online softmax - computes max and sum in a single pass.

    Instead of:
        Pass 1: find max
        Pass 2: compute sum(exp(x - max))
        Pass 3: divide

    We do:
        Pass 1: find max AND sum together (rescale sum when new max found)
        Pass 2: divide

    When a new max m' is found, we rescale the running sum:
        old: s = Σ exp(x_i - m)
        new: s' = s * exp(m - m') + 1

    This works because:
        Σ exp(x_i - m') = Σ exp(x_i - m) * exp(m - m')

    Passes: 2 (max+sum, divide)
    """
    m = -np.inf  # running max
    s = 0.0      # running sum

    # Single pass: track max and sum together
    for xi in x:
        if xi > m:
            # Found new max, rescale previous sum
            s = s * np.exp(m - xi) + 1.0
            m = xi
        else:
            s = s + np.exp(xi - m)

    # Final pass: compute softmax
    return np.exp(x - m) / s


def test():
    # Example usage
    x = np.array([0, 1001, 1002])  # Large values that would overflow

    print("Testing with large values:", x)
    try:
        result = softmax_unstable(x)
        print("Unstable softmax:", result)
    except Exception as e:
        print("Unstable softmax error:", str(e))
    
    result_1 = softmax_stable(x)
    result_2 = softmax_online(x)
    print("Stable softmax:", result_1)
    print("Online softmax:", result_2)
    assert np.allclose(result_1, result_2)

if __name__ == "__main__":
    test()
