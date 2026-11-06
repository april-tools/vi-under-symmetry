import torch
import torch.nn as nn
from pyro.distributions import MultivariateStudentT

class FCTProposal(nn.Module):
    def __init__(self,
                 mean,
                 scale,
                 df: int,
                 device: str,
                 scale_learnable: bool):
        """
        Full covariance Student-t variational family with df degrees of freedom.
        mean: (D, ) tensor
        scale: unconstrained (D, D) tensor that will get transformed into the scale matrix
        df: degrees of freedom
        """
        super().__init__()

        ### properties of the family relevant to the theorems
        self.is_log_concave = False
        self.is_log_concave_strict = False
        self.is_rad_incr_strict = True

        D = mean.shape[0]
        self.D = D
        self.mean = nn.Parameter(mean.to(device))
        self.scale = nn.Parameter(scale.to(device))
        self.df = torch.tensor(df, device=device, requires_grad=False, dtype=self.mean.dtype)
        assert self.df > 2 # ensure finite first and second moment
        self.device = device

        # enable gradients
        self.mean.requires_grad = True
        self.scale.requires_grad = scale_learnable
        self.df.requires_grad = False
        print("Scale learnable:", self.scale.requires_grad)

    def _get_cholesky(self):
        # transform unconstrained matrix to lower triangular with pos diag -> valid cholesky
        L = torch.tril(self.scale) # L is learned
        diag_raw = torch.diagonal(L)
        diag_pos = torch.exp(diag_raw)
        L = L - torch.diag(diag_raw) + torch.diag(diag_pos)
        return L

    def _dist(self):
        L = self._get_cholesky()
        return MultivariateStudentT(
            df=self.df,
            loc=self.mean,
            scale_tril=L
        )

    def sample(self, num_samples=1):
        return self._dist().rsample((num_samples,))  # reparameterized


    def log_p(self, x):
        return self._dist().log_prob(x)

    @property
    def cov(self):
        L = self._get_cholesky()
        cov = (self.df / (self.df - 2)) * (L @ L.T)
        return cov

    @property
    def corr(self):
        cov = self.cov
        std = torch.sqrt(torch.diag(cov)) # marginal standard deviations
        corr = cov / torch.outer(std, std)
        return corr

    @property
    def L(self):
        return self._get_cholesky()