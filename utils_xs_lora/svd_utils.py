from sklearn.decomposition import TruncatedSVD
import numpy as np
from typing import Tuple
import torch


def run_svd(input_matrix: np.ndarray, rank: int, n_iter: int, random_state: int) -> Tuple[np.ndarray, TruncatedSVD]:
    svd = TruncatedSVD(n_components=rank, n_iter=n_iter, random_state=random_state)
    svd.fit(input_matrix)
    reduced_matrix = svd.transform(input_matrix)
    return reduced_matrix, svd


def get_linear_rec_svd(input_matrix: np.ndarray, rank: int, n_iter: int,
                       random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced_matrix, svd = run_svd(input_matrix, rank, n_iter, random_state)

    reconstructed_matrix = svd.inverse_transform(reduced_matrix)
    return reconstructed_matrix, reduced_matrix, svd.components_ 

# svd.components_ is V_r^T
# reduced_matrix is U_r Σ_r


@torch.no_grad()
def svd_lowrank(W: torch.Tensor, rank: int, sigma_split: str = "sym", eps: float = 1e-12):
    """
    W: (out_dim, in_dim) torch.Tensor
    rank: desired rank (will be clipped to min(out_dim, in_dim))
    sigma_split: "sym" | "left" | "right"
      - "sym":  A = U * sqrt(S),   B = sqrt(S) * Vh
      - "left": A = U * S,         B = Vh
      - "right":A = U,             B = S * Vh
    eps: floor for tiny singular values to avoid sqrt underflow in low-precision dtypes
    """
    assert W.ndim == 2, f"weight must be 2D, got {tuple(W.shape)}"
    m, n = W.shape
    r = min(rank, m, n)
    if r <= 0:
        raise ValueError(f"rank must be > 0, got {rank}")

    # Always factorize in fp32 for numerical stability.
    W32 = W.detach().to(torch.float32).cpu()

    U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
    U = U[:, :r]            # (m, r)
    S = S[:r]               # (r,)
    Vh = Vh[:r, :]          # (r, n)
    
    print(U, S, Vh)

    # Avoid sqrt(0) and tiny underflow in half/bfloat16 later.
    S_safe = torch.clamp(S, min=eps)
    sqrtS = torch.sqrt(S_safe)  # (r,)

    if sigma_split == "sym":
        # A = U @ diag(sqrtS)  -> broadcast multiply columns of U by sqrtS
        # B = diag(sqrtS) @ Vh -> broadcast multiply rows of Vh by sqrtS
        A = U * sqrtS   # (m, r)
        B = sqrtS.unsqueeze(1) * Vh  # (r, n)
    elif sigma_split == "left":
        A = U * S       # (m, r)
        B = Vh          # (r, n)
    elif sigma_split == "right":
        A = U
        B = S.unsqueeze(1) * Vh
    else:
        raise ValueError(f"Unknown sigma_split: {sigma_split}")

    # Optional: re-scale so ||A@B||_F matches ||W||_F (helps if many tiny S)
    recon_norm = torch.linalg.norm(A @ B)
    W_norm = torch.linalg.norm(W32)
    if recon_norm > 0:
        scale = (W_norm / recon_norm).sqrt()  # split scale symmetrically
        A = A * scale
        B = B * scale

    return A, B