"""SVD from scratch via the eigendecomposition of A^T A.

Identity used:  A^T A = V Σ² V^T   (symmetric, so use torch.linalg.eigh)
  -> right singular vectors V = eigenvectors of A^T A
  -> singular values        σ = sqrt(eigenvalues)
  -> left  singular vectors U = A V / σ   (cheap, sign-consistent with V)
"""

import torch


def svd_from_eig(A: torch.Tensor):
    """Compute SVD of A (shape m x n) so that A = U @ diag(S) @ Vh.

    Returns:
        U  (m x k), S (k,), Vh (k x n)   where k = min(m, n)
    """
    m, n = A.shape

    eigvals, V = torch.linalg.eigh(A.T @ A)

    S, U, Vh = reconstruct(A, eigvals, V, m, n)
    return U, S, Vh


def reconstruct(A, eigvals, V, m, n):
    """TODO(you): implement the reordering + U construction described above.

    Inputs:
        A        : original matrix (m x n)
        eigvals  : eigenvalues of A^T A, ASCENDING (n,)
        V        : eigenvectors of A^T A, columns matching eigvals (n x n)
    Returns:
        S  : singular values, DESCENDING (k,)
        U  : left singular vectors (m x k)
        Vh : right singular vectors, transposed (k x n)
        where k = min(m, n)
    """
    k = min(m, n)

    idx = torch.argsort(eigvals, descending=True)[:k]
    eigvals = eigvals[idx]
    V = V[:, idx]
    S = eigvals.clamp(min=0).sqrt()

    tol = 1e-10
    safe_S = torch.where(S > tol, S, torch.ones_like(S))
    AV = A @ V
    U = torch.where(S > tol, AV / safe_S, torch.zeros_like(AV))
    return S, U, V.T


def demo(name: str, A: torch.Tensor):
    U, S, Vh = svd_from_eig(A)
    A_rec = U @ torch.diag(S) @ Vh

    print(f"\n=== {name}  (shape {tuple(A.shape)}) ===")
    print("A =\n", A)
    print("singular values:", S)
    print("reconstruction max error:", (A - A_rec).abs().max().item())

    # cross-check against PyTorch's built-in
    U2, S2, Vh2 = torch.linalg.svd(A, full_matrices=False)
    print("matches torch.linalg.svd singular values:",
          torch.allclose(S, S2, atol=1e-5))


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    # 1. a simple square matrix
    demo("square", torch.tensor([[4.0, 0.0],
                                 [3.0, -5.0]]))

    # 2. a tall, rank-deficient matrix (rank 1: col2 = 2*col1)
    demo("tall, rank-deficient", torch.tensor([[1.0, 2.0],
                                               [2.0, 4.0],
                                               [3.0, 6.0]]))

    # 3. a wide random matrix
    demo("wide random", torch.randn(2, 4))
