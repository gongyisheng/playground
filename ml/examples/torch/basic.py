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

def test_contiguous():
    # operations can break contiguity
    # transpose, permute, narrow, expand, view, tensor[::2]
    
    # need to be contiguous before
    # view/reshape, .numpy(), or other cuda operations
    
    q = torch.rand(2, 3, 4)
    print("Original tensor, is contiguous:", q.is_contiguous())

    q_transposed = q.transpose(1, 2)
    print("After transpose, is contiguous:", q_transposed.is_contiguous())

    q_contiguous = q_transposed.contiguous()
    print("After contiguous, is contiguous:", q_contiguous.is_contiguous())


if __name__ == "__main__":
    # test_transpose()
    # test_mean()
    # test_variance()
    test_contiguous()