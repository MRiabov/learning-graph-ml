import torch
from torch.nn import functional as F


def mpm_batch(K, num_iters=75, eps=1e-8):
    """
    Args:
        K - affinity matrix, shape (B, N2, N2)
    Returns:
        x: Soft assignment matrix, shape (B, N2)
    """
    B, n2, _ = K.shape
    x = torch.full(
        (B, n2), 1.0 / n2, device=K.device, dtype=K.dtype
    )  # prob matrices split equally.

    for _ in range(num_iters):
        x = torch.bmm(K, x.unsqueeze(-1)).squeeze(-1)  # torch.bmm requires 3-dim
        # x = torch.vmap(torch.matmul)(K, x) # matmul
        x = x.clamp(min=eps)  # avoid divide by zero.
        x = x / x.sum()  # l1 normalization

    return x


def mpm_to_perm_batch(x, n: int):
    """
    Args:
        x: Soft assignment matrix of shape (B, n^2, n^2)
        n: Number of nodes.

    Returns:
        Permutation matrix of shape (n, n)
    """
    B = x.shape[0]

    x_mat = x.view(B, n, n)  # soft assignment matrix
    row_ind = x_mat.argmax(dim=1)  # find max index per row
    perm = F.one_hot(row_ind, num_classes=n).to(dtype=x_mat.dtype, device=x_mat.device)
    return perm


def build_affinity_matrix(node_logits, node_classes_y):
    B, N, C = node_logits.shape
    node_classes_y_one_hot = F.one_hot(
        node_classes_y, num_classes=C
    ).float()  # (B, N, C)

    # compute node affinity (A_pred@A_true^T)
    S = torch.bmm(node_logits, node_classes_y_one_hot.transpose(1, 2))  # (B, N, N)

    # Flatten rows and columns
    S_flat = S.view(B, -1)  # (B, N^2)
    K = torch.bmm(S_flat.unsqueeze(2), S_flat.unsqueeze(1))

    return K
