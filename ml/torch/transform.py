import torch


def test_transpose():
    q = torch.rand(1,4,3)
    print("Before transpose:")
    print(q.shape) # [1,4,3]
    print(q)

    q = q.transpose(-1,-2)
    print("After transpose:")
    print(q.shape) # [1,3,4]
    print(q)

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
    test_transpose()
    # test_contiguous()
