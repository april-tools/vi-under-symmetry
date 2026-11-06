"""
Evaluation metrics
"""
import torch

def delta_mean(p, q):
    return torch.mean(torch.abs(p.mean - q.mean)/torch.max(torch.sqrt(torch.diag(p.cov)), torch.abs(p.mean)))


def delta_corr(p, q):
    diff = torch.abs(p.corr - q.corr)
    lower = torch.tril(diff, diagonal=-1) # omit diagonal
    result = 2 * lower.sum() / (diff.shape[0] * (diff.shape[0] - 1)) # 2 * since symmetric
    return result