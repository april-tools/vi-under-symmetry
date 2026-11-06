"""
Asymmetric GMM targets used in experiments and figures.
"""

import torch
from .gmm import GMMTarget

class AGMMTarget2D(GMMTarget):
    """
    A-GMM Target used in experiments.
    """
    def __init__(self, device: str):
        m = torch.tensor([[5, 4], [-2, -6]]).float()
        var = torch.tensor([[1, 2], [0.5, 1]]).float()
        corr = torch.tensor([[[1, -0.4], [-0.4, 1]], [[1, 0.2], [0.2, 1]]]).float()
        super().__init__(m, var, corr, device=device, is_sym=False)

class AGMMTarget2DFig(GMMTarget):
    """
    Used for Figure 2.
    """
    def __init__(self, device: str):
        K = 2
        m = torch.tensor([3, 3]).float()
        mean = m.unsqueeze(0).repeat(K, 1)
        mean[-1, :] = -mean[-1, :]  # symmetry
        mean[-1, :] = mean[-1, :] + torch.tensor([0.5, -0.2])
        m = mean
        var = torch.tensor([1, 2]).float()
        var = var.unsqueeze(0).repeat(K, 1)
        corr = torch.tensor([[1, 0.2], [0.2, 1]]).float()
        corr = corr.unsqueeze(0).repeat(K, 1, 1)
        corr[-1,:] = torch.tensor([[1, -0.4], [-0.4, 1]]).float()
        super().__init__(m, var, corr, device=device, is_sym=False)