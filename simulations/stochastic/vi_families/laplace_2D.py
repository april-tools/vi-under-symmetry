import torch.nn as nn
from torch.special import modified_bessel_k0, modified_bessel_k1
from torch.distributions import Exponential, MultivariateNormal

import torch

class BesselK0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return modified_bessel_k0(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = -modified_bessel_k1(x)
        return grad_output * grad


class FCLProposal(nn.Module):
    def __init__(self,
                 mean,
                 scale,
                 device: str,
                 scale_learnable: bool):
        """
        Full covariance Laplace variational family.
        mean: (D, ) tensor
        scale: unconstrained (D, D) tensor that will get transformed into the scale matrix
        """

        super().__init__()

        ### properties of the family relevant to the theorems
        self.is_log_concave = False
        self.is_log_concave_strict = False
        self.is_rad_incr_strict = True

        D = mean.shape[0]
        assert D == 2
        self.v = (2-D)/2
        self.device = device
        self.exp_dist = Exponential(torch.tensor(1.0, device=self.device))

        self.D = D
        self.mean = nn.Parameter(mean.to(device))
        self.scale = nn.Parameter(scale.to(device))

        # ensure grad
        self.mean.requires_grad = True
        self.scale.requires_grad = scale_learnable

        print("Scale learnable:", self.scale.requires_grad)

    def _get_cholesky(self):
        L = torch.tril(self.scale)
        diag_raw = torch.diagonal(L)
        diag_pos = torch.exp(diag_raw) # exp the diag
        L = L - torch.diag(diag_raw) + torch.diag(diag_pos)
        return L

    def sample(self, num_samples=1):
        W = self.exp_dist.rsample((num_samples,))
        Y = MultivariateNormal(loc=torch.zeros(self.D, device=self.device),
                               covariance_matrix=self.cov).rsample((num_samples, ))
        return self.mean + torch.sqrt(W)[:, None] * Y

    def log_p(self, x):
        # log-pdf of the multivariate Laplace
        L = self._get_cholesky()
        sigma = L@L.T # covariance matrix
        sigma_inv = torch.linalg.inv(sigma)
        sign, logdet = torch.linalg.slogdet(sigma) # p.d.
        x_c = x - self.mean
        x_maha = torch.sqrt(2*torch.sum((x_c @ sigma_inv) * x_c, dim=1))
        return -torch.log(torch.tensor(torch.pi)) - 0.5 * logdet + torch.log(BesselK0.apply(x_maha))

    @property
    def corr(self):
        cov = self.cov
        std = torch.sqrt(torch.diag(cov)) # marginal standard deviations
        corr = cov / torch.outer(std, std)
        return corr

    @property
    def cov(self):
        L = self._get_cholesky()
        cov = L @ L.T
        return cov

    @property
    def L(self):
        return self._get_cholesky()