import torch

def fem_residual_loss(u_pred, K, F):
    u = u_pred.unsqueeze(-1)          # (B, 39, 1)
    R = torch.bmm(K, u) - F           # (B, 39, 1)
    norm_R = torch.linalg.vector_norm(R, ord=2)
    norm_F = torch.linalg.vector_norm(F, ord=2)
    loss = norm_R / (norm_F + 1e-6)
    return loss
