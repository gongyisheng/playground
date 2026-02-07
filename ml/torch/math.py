import torch


def test_mean():
    q = torch.rand(2,2,5)
    mean_q = q.mean(dim=-1, keepdim=False)
    print("keepdim=False:")
    print(mean_q.shape) # [2,2]
    print(mean_q)

    mean_q = q.mean(dim=-1, keepdim=True)
    print("keepdim=True:")
    print(mean_q.shape) # [2,2,1]
    print(mean_q)

def test_variance():
    q = torch.rand(2,5)
    var_q = q.var(dim=-1, keepdim=True, unbiased=True) # divide by n-1, works with sample (estimation)
    print("unbiased=True:")
    print(var_q.shape)
    print(var_q)

    var_q = q.var(dim=-1, keepdim=True, unbiased=False) # divide by n, works with population variance
    print("unbiased=False:")
    print(var_q.shape)
    print(var_q)

def test_inplace():
    # trailing _ means in-place: modifies the tensor directly, no new tensor created
    x = torch.tensor([1.0, 2.0, 3.0])
    print("original:", x)

    x.add_(10.0)       # x += 10
    print("add_(10):", x)

    x.mul_(2.0)        # x *= 2
    print("mul_(2):", x)

    x.zero_()           # x = 0
    print("zero_():", x)

    x.fill_(7.0)        # x = 7
    print("fill_(7):", x)

    # not in-place: returns a new tensor, original unchanged
    y = torch.tensor([1.0, 2.0, 3.0])
    z = y.add(10.0)
    print("y (unchanged):", y)
    print("z (new tensor):", z)

if __name__ == "__main__":
    test_mean()
    # test_variance()
    # test_inplace()
