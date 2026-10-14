# Numerical overflow
import numpy as np

# Sigmoid function
# can be written in two ways:
# 1. 1/(1+exp(-x))
# 2. exp(x)/(1+exp(x))
# Each one takes care of overflow for big positive and negative numbers
# Need to combine them to avoid overflow

def sigmod_unstable(x):
    return 1/(1+np.exp(-x))

def sigmod_stable(x):
    if x < 0:
        return np.exp(x)/(1+np.exp(x))
    else:
        return 1/(1+np.exp(-x))

def test():
    # Example usage
    n = -100000000000000000
    try:
        print(sigmod_unstable(n))  # This may cause overflow
    except Exception as e:
        print(str(e))
    print(sigmod_stable(n))  # This should work fine

if __name__ == "__main__":
    test()