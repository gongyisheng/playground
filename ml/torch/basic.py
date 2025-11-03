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

if __name__ == "__main__":
    test_transpose()