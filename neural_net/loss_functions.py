import torch

def fem_residual_loss(u_pred, K, F):
    u = u_pred.unsqueeze(-1)          # (B, 39, 1)
    Ku = torch.bmm(K, u)              # (B, 39, 1)
    residuo = Ku - F                  # (B, 39, 1)
    norma_ao_quadrado = torch.sum(residuo**2, dim=1) / (torch.sum(F**2, dim=1) + 1e-8)
    loss = torch.mean(norma_ao_quadrado)
    return loss
