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


def svd_lowrank(W: torch.Tensor, rank: int):
    """
    Low-rank factorization of weight matrix W for LoRA-style update:
    delta = x @ A @ L @ B
    where A and B come from truncated SVD of W.
    """
    # Full SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # Truncate to rank
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # Distribute Σ symmetrically across A and B
    A = U_r @ torch.diag(torch.sqrt(S_r))     # (out_dim, rank)
    B = torch.diag(torch.sqrt(S_r)) @ Vh_r    # (rank, in_dim)

    return A, B

# svd.components_ is V_r^T
# reduced_matrix is U_r Σ_r