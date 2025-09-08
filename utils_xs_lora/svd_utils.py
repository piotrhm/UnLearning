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
def svd_lowrank(W: torch.Tensor, rank: int, split_sigma: str = "left"):
    """
    Low-rank factorization of W into A, B for LoRA-style update.
    split_sigma:
      - "sym": split sqrt(S) across A and B  (default, balanced)
      - "left": put all S in A
      - "right": put all S in B
    """
    # Always work in fp32 for stability
    W = W.to(torch.float32)

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :rank], S[:rank], Vh[:rank, :]

    if split_sigma == "sym":
        A = U_r @ torch.diag(torch.sqrt(S_r))
        B = torch.diag(torch.sqrt(S_r)) @ Vh_r
    elif split_sigma == "left":
        A = U_r @ torch.diag(S_r)
        B = Vh_r
    elif split_sigma == "right":
        A = U_r
        B = torch.diag(S_r) @ Vh_r
    else:
        raise ValueError(f"Unknown split_sigma: {split_sigma}")

    return A, B