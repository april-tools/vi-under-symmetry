"""
Symmetric GMM targets used in experiments and figures.
"""

import torch
import torch.nn as nn

class GMMTarget(nn.Module):
    def __init__(self,
                 mean,
                 var,
                 corr,
                 device,
                 weights=None,
                 is_sym=True):
        """
        mean:   (K, D)
        var:    (K, D) marginal variances
        corr:   (K, D, D) correlation structure
        weight: (K, ) # mixture weights
        """
        super().__init__()
        self.is_even = is_sym
        self.is_elliptic = False
        self.is_log_concave = False
        self.support_cond = True

        K, D = mean.shape
        self.D = D
        self.K = K
        self.device = device
        if weights is None:
            print("Using equal weighting for mixture components, \n")
            weights = torch.ones(K)*(1/K)
        assert var.shape == (K, D)
        assert corr.shape == (K, D, D)
        assert weights.shape == (K,)
        assert torch.allclose(torch.sum(weights, dim=-1), torch.tensor(1.0))

        # detach (no gradients)
        self.means = mean.detach().float().to(self.device)
        self.vars = var.detach().float().to(self.device)
        self.corrs = corr.detach().float().to(self.device)
        self.weights = weights.detach().float().to(self.device)

        # build covariance matrices (K, D, D)
        std = torch.sqrt(self.vars)  # (K, D) marginal standard deviations
        var_mats = torch.diag_embed(std) # (K, D, D)
        self.covs = var_mats @ self.corrs @ var_mats # (K, D, D)
        print("component means", self.means)
        print("component covs:", self.covs)

        # sanity check
        assert torch.allclose(torch.diagonal(self.covs, dim1=-2, dim2=-1), self.vars)

        # component distribution (batch of K Gaussians)
        comp = torch.distributions.MultivariateNormal(
            loc=self.means,
            covariance_matrix=self.covs
        )

        # mixing distribution
        mix = torch.distributions.Categorical(self.weights)

        # final mixture
        self.dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=mix,
            component_distribution=comp
        )

    def sample(self, num_samples):
        return self.dist.sample((num_samples,))

    def log_p(self, x):
        return self.dist.log_prob(x)

    @property
    def mean(self):
        # computes mixture mean via law of total expectation
        return torch.sum(self.weights[:, None] * self.means, dim=0)

    @property
    def cov(self):
        """
        Computes mixture covariance via law of total variance
        """
        K, D = self.means.shape

        # component covariances
        covs_k = self.covs # component covs

        # mixture mean
        mean = self.mean  # (D,)

        # compute mixture covariance
        total_cov = torch.zeros(D, D, device=self.means.device)

        for k in range(K):
            diff = (self.means[k] - mean).unsqueeze(1)  # (D, 1)
            total_cov += self.weights[k] * (covs_k[k] + diff @ diff.T)

        return total_cov

    @property
    def corr(self):
        cov = self.cov
        std = torch.sqrt(torch.diag(cov)) # marginal stddevs

        # outer product of stds
        denom = torch.outer(std, std)

        return cov / denom


class GMMTarget2D(GMMTarget):
    def __init__(self, device: str):
        K = 2
        m = torch.tensor([5, 5]).float()
        mean = m.unsqueeze(0).repeat(K, 1) # symmetry!
        mean[-1,:] = -mean[-1, :] # symmetry
        mean = mean + 2.0 # shift the mean away from origin but retain symmetry
        var = torch.tensor([1, 2]).float()
        var = var.unsqueeze(0).repeat(K, 1)
        corr = torch.tensor([[1, -0.4], [-0.4, 1]]).float()
        corr = corr.unsqueeze(0).repeat(K, 1, 1)
        super().__init__(mean, var, corr, device=device)


class GMMTarget4D(GMMTarget):
    def __init__(self, device: str):
        K = 2
        D = 4
        m = torch.ones(D).float() * 1.0
        m[::2] = 1
        mean = m.unsqueeze(0).repeat(K, 1)  # symmetry!
        mean[-1, :] = -mean[-1, :]  # symmetry
        mean = mean + 1.0  # shift the mean away from origin but retain symmetry

        cov = torch.zeros(D, D)
        cov.fill_diagonal_(1)
        std = torch.sqrt(torch.diag(cov))
        corr = cov / torch.outer(std, std)
        corr = corr.unsqueeze(0).repeat(K, 1, 1)
        vars = torch.diag(cov).repeat(K, 1)
        super().__init__(mean, vars, corr, device=device, is_sym=True)


class GMMTarget8D(GMMTarget):
    def __init__(self, device: str):
        K = 2
        D = 8
        m = torch.ones(D).float() * 1.0
        m[::2] = 1
        mean = m.unsqueeze(0).repeat(K, 1)  # symmetry!
        mean[-1, :] = -mean[-1, :]  # symmetry
        mean = mean + 1.0 # shift the mean away from origin but retain symmetry

        cov = torch.zeros(D, D)
        cov.fill_diagonal_(1)
        std = torch.sqrt(torch.diag(cov))
        corr = cov / torch.outer(std, std)
        corr = corr.unsqueeze(0).repeat(K, 1, 1)
        vars = torch.diag(cov).repeat(K, 1)
        super().__init__(mean, vars, corr, device=device, is_sym=True)


class GMMTarget16D(GMMTarget):
    def __init__(self, device: str):
        K = 2
        D = 16
        m = torch.ones(D).float() * 1.0
        m[::2] = 1
        mean = m.unsqueeze(0).repeat(K, 1)  # symmetry!
        mean[-1, :] = -mean[-1, :]  # symmetry
        mean = mean + 1.0  # shift the mean away from origin but retain symmetry

        cov = torch.zeros(D, D)
        cov.fill_diagonal_(1)
        std = torch.sqrt(torch.diag(cov))
        corr = cov / torch.outer(std, std)
        corr = corr.unsqueeze(0).repeat(K, 1, 1)
        vars = torch.diag(cov).repeat(K, 1)
        super().__init__(mean, vars, corr, device=device, is_sym=True)


class GMMTarget2DFig(GMMTarget):
    """
    Used for Figure 2.
    """
    def __init__(self, device: str):
        K = 2
        m = torch.tensor([3, 3]).float()
        mean = m.unsqueeze(0).repeat(K, 1) # symmetry!
        mean[-1,:] = -mean[-1, :] # symmetry
        var = torch.tensor([1, 2]).float()
        var = var.unsqueeze(0).repeat(K, 1)
        corr = torch.tensor([[1, 0.2], [0.2, 1]]).float()
        corr = corr.unsqueeze(0).repeat(K, 1, 1)
        super().__init__(mean, var, corr, device=device)






