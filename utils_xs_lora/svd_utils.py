import torch

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