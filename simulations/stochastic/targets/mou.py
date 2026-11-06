"""
2D mixture of uniforms used in experiments.
"""

import torch

class MixtureOfUniforms:
    def __init__(self,
                 lows,
                 highs,
                 device="cpu"):
        """
        weights: (K,) mixture weights
        lows: (K, D) lower bounds per component and dimension
        highs: (K, D) upper bound per component and dimension
        """

        self.is_even = True
        self.is_elliptic = False
        self.is_log_concave = False
        self.support_cond = False

        self.device = device

        mixture_weights = torch.ones(lows.shape[0]) / lows.shape[0] # uniform weights

        self.weights = mixture_weights.to(device)
        self.cat = torch.distributions.Categorical(self.weights)
        self.lows = lows.to(device)
        self.highs = highs.to(device)

        self.K = lows.shape[0] # number of components
        self.D = lows.shape[1] # number of dimensions

        # pre-compute the density value on each component
        self.densities = 1/torch.prod(self.highs - self.lows, dim=-1)

    def log_p(self, x):
        x = x.unsqueeze(1) # (N, 1, D)

        # check whether the point lies inside the support of either of the component
        in_interval = (x >= self.lows) & (x <= self.highs)
        in_box = in_interval.all(dim=-1)
        densities = self.densities.unsqueeze(0).expand(x.shape[0], -1) # density per component

        # compute the density x receives from each component
        densities = torch.where(
            in_box,
            densities,
            torch.zeros_like(densities)
        )

        weighted = densities * self.weights
        pdf = weighted.sum(dim=-1)

        return torch.log(pdf)

    def sample(self, n):
        # sample the component
        indices = self.cat.sample((n,))
        lows = self.lows[indices]
        highs = self.highs[indices]

        # sample a U(0, 1) RV
        u = torch.rand(n, self.D).to(self.device)
        # map it to the interval
        samples = lows + (highs - lows) * u

        return samples

    @property
    def mean(self):
        comp_means = (self.lows + self.highs) / 2
        mean = (self.weights.unsqueeze(-1) * comp_means).sum(dim=0)
        return mean

    @property
    def cov(self):
        comp_means = (self.lows + self.highs) / 2  # mean per component (K, D)

        # variance of each uniform component: (b - a)^2 / 12
        comp_vars = (self.highs - self.lows) ** 2 / 12.0  # (K, D)

        comp_covs = torch.diag_embed(comp_vars)  # (K, D, D), marginal variances

        mean = self.mean  # (D,) # overall mean

        # E[Cov(X|Z)]
        term1 = (self.weights.view(-1, 1, 1) * comp_covs).sum(dim=0)

        # Cov(E[X|Z])
        mean_diffs = comp_means - mean  # (K, D)
        outer = mean_diffs.unsqueeze(-1) * mean_diffs.unsqueeze(-2)  # (K, D, D)
        term2 = (self.weights.view(-1, 1, 1) * outer).sum(dim=0)

        return term1 + term2

    @property
    def corr(self):
        """
        returns: (D, D)
        """
        cov = self.cov
        std = torch.sqrt(torch.diag(cov))
        denom = std.unsqueeze(0) * std.unsqueeze(1)

        corr = cov / denom
        return corr

class MOU(MixtureOfUniforms):
    def __init__(self, device):
        direction = torch.tensor([1.0, -0.3]).to(device)
        direction = direction / direction.norm()  # normalize
        shift = 2.0

        distance = 6.0
        loc = distance * direction

        scale = torch.tensor([2, 2]).to(device)

        highs_1 = loc + scale + shift
        lows_1 = loc - scale + shift

        highs_2 = -loc + scale + shift
        lows_2 = -loc - scale + shift

        lows = torch.zeros((2, 2)).to(device)
        lows[0,:] = lows_1
        lows[1,:] = lows_2

        highs = torch.zeros((2, 2)).to(device)
        highs[0, :] = highs_1
        highs[1, :] = highs_2

        print("c_1 x_1", lows[0, 0], highs[0, 0], "c_1 x_2", lows[0, 1], highs[0, 1], "c_2 x_1", lows[1, 0],
              highs[1, 0], "c_2 x_2", lows[1, 1], highs[1, 1])
        super().__init__(lows, highs, device)
