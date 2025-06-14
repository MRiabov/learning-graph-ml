import torch
from torch.nn import functional as F


def mpm_batch(K, num_iters=75, eps=1e-8):
    _B, n2 = K.shape
    x = torch.full_like(K, 1.0 / n2)  # prob matrices split equally.

    for _ in range(num_iters):
        x = K * x  #FIXME: this approximation may not be correct.
        # x = torch.vmap(torch.matmul)(K, x) # matmul
        x = x.clamp(min=eps)  # avoid divide by zero.
        x = x / x.sum()  # l1 normalization

    return x


def mpm_to_perm_batch(x, n: int):
    """
    Args:
        x: Soft assignment matrix of shape (n^2,)
        n: Number of nodes.

    Returns:
        Permutation matrix of shape (n, n)
    """
    B = x.shape[0]

    x_mat = x.view(B, n, n)  # x matrix, I presume.
    row_ind = x_mat.argmax(dim=1)  # find max index per row
    perm = F.one_hot(row_ind, num_classes=n).to(dtype=x_mat.dtype, device=x_mat.device)
    return perm


def build_affinity_matrix(node_logits, node_classes_y):
    B, N, C = node_logits.shape
    gt_node_feats = F.one_hot(node_classes_y, num_classes=C).float()  # (B, N, C)
    S = torch.bmm(node_logits, gt_node_feats.transpose(1, 2))  # (B, N, N)
    K = S.view(B, -1)
    return K
