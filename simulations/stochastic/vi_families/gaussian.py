import torch
import torch.nn as nn


class FCGProposal(nn.Module):
    def __init__(self, mean,
                 scale,
                 device: str,
                 scale_learnable: bool):
        """
        Full covariance Gaussian variational family.
        mean: (D, ) tensor
        scale: unconstrained (D, D) tensor that will get transformed into the scale matrix
        """
        super().__init__()

        ### properties of the family relevant to the theorems
        self.is_log_concave = True
        self.is_log_concave_strict = True
        self.is_rad_incr_strict = True

        D = mean.shape[0]
        self.D = D
        self.mean = nn.Parameter(mean.to(device))
        self.scale = nn.Parameter(scale.to(device))
        self.device = device

        # ensure grad
        self.mean.requires_grad = True
        self.scale.requires_grad = scale_learnable

        print("Scale learnable:", self.scale.requires_grad)

    def _get_cholesky(self):
        """
        Compute Cholesky factor L such that the scale will be given by LL^T.
        """
        L = torch.tril(self.scale) # parameter that is learned
        diag_raw = torch.diagonal(L)
        diag_pos = torch.exp(diag_raw) # ensure positive diagonal
        L = L - torch.diag(diag_raw) + torch.diag(diag_pos)
        return L

    def _dist(self):
        L = self._get_cholesky()
        return torch.distributions.MultivariateNormal(
            loc=self.mean,
            scale_tril=L
        )

    def sample(self, num_samples=1):
        return self._dist().rsample((num_samples,))  # reparameterized sampling

    def log_p(self, x):
        return self._dist().log_prob(x)

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

