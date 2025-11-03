import torch

def test_transpose():
    q = torch.rand(1,4,3)
    print("Before transpose:")
    print(q.shape)
    print(q)

    q = q.transpose(-1,-2)
    print("After transpose:")
    print(q.shape)
    print(q)

def test_mean():
    q = torch.rand(2,5)
    mean_q = q.mean(dim=-1, keepdim=False)
    print("keepdim=False:")
    print(mean_q.shape)
    print(mean_q)

    mean_q = q.mean(dim=-1, keepdim=True)
    print("keepdim=True:")
    print(mean_q.shape)
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

if __name__ == "__main__":
    # test_transpose()
    # test_mean()
    test_variance()