"""
Ellipse targets build using the cirkit library.
"""

import torch


import cirkit.symbolic.functional as SF
from cirkit.pipeline import PipelineContext

torch.set_default_dtype(torch.float64)

from collections import defaultdict
from cirkit.symbolic.circuit import Circuit
from cirkit.symbolic.initializers import (
    ConstantTensorInitializer,
)
from cirkit.symbolic.layers import GaussianLayer, SumLayer, HadamardLayer
from cirkit.templates.utils import (
    Scope,
)
from cirkit.symbolic.parameters import (
    Parameter,
    TensorParameter,
)

from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily

def set_ellipse_params(D):
    """
    Define the parameters of the baseRing
    :param D:
    :return:
    """
    m_off_diag = 2
    m_diag = 4
    mean_val = 1
    mu = torch.ones(D) * mean_val
    M = torch.ones((D, D)) * m_off_diag
    M.fill_diagonal_(m_diag)
    mean = torch.zeros(D, 2)
    var = torch.ones(D, 2)
    var[:, 1] = 0.3
    base_params = {
        "mean": mean,
        "stddev": torch.sqrt(var),
        "weights": torch.tensor([1 - 1 / (D ** 2), -1 / (D ** 2)])
    }
    return base_params, mu, M

def build_PC(
    weights: dict, n_features: int , n_comps: int
):
    """
    Build a circuit encoding the target before squaring
    :param weights: parameter dict: "mean" (D, 2), "stddev" (D, 2), "weights" (1,2)
    :param n_features: number of features (2)
    :param n_comps: number of components (2)
    :return:
    """
    layers = []
    input_layers = []
    in_layers_dict = defaultdict(list)

    for i in range(n_features):  # layer per variable
        mean = Parameter.from_input(
            TensorParameter(
                *(n_comps,),
                initializer=ConstantTensorInitializer(
                    weights["mean"][i, :].detach().numpy() # mean per component
                ),
                learnable=False,
            )
        )

        stddev = Parameter.from_input(
            TensorParameter(
                *(n_comps,),
                initializer=ConstantTensorInitializer(
                   weights["stddev"][i, :].numpy() # stddev per component
                ),
                learnable=False,
            )
        )

        input_layers.append(
            GaussianLayer(
                scope=Scope({i}),
                num_output_units=weights["mean"].shape[1], # number of components
                mean=mean,
                stddev=stddev,
            )
        )
    layers.extend(input_layers)

    # add a Hadamard layer to combine distributions over variables
    product_layer = HadamardLayer(num_input_units=n_comps, arity=n_features)
    in_layers_dict[product_layer] = input_layers
    layers.append(product_layer)

    # add a sum layer to create the mixture (shape: (1, n_comps))
    if len(weights["weights"].shape) == 1:
        weights["weights"] = weights["weights"].unsqueeze(0)

    weight = Parameter.from_input(
        TensorParameter(
            *((1, weights["weights"].shape[1])), # numbe of components
            initializer=ConstantTensorInitializer(
                weights["weights"].numpy()
            ),
            learnable=False

        )
    )

    sum_layer = SumLayer(num_input_units=n_comps, num_output_units=1, weight=weight)
    in_layers_dict[sum_layer] = [product_layer]
    layers.append(sum_layer)

    # build the symbolic circuit
    circuit = Circuit(
        layers=layers, in_layers=in_layers_dict, outputs=[sum_layer]
    )
    return circuit


class Ring:
    """
    Constructs a spherical base distribution.
    """

    def __init__(self, device, circuit_params, D):
        super(Ring, self).__init__()
        self.D = D
        self.device = device
        self.circuit_params = circuit_params
        self.build_circuit()

    def build_circuit(self):
        ctx = PipelineContext(
            backend="torch", semiring="complex-lse-sum", fold=True, optimize=False
        )
        # build the symbolic circuit
        symbolic_circuit = build_PC(
            self.circuit_params, n_comps=2, n_features=self.D
        )

        # square the circuit to ensure non-negative density
        symbolic_circuit = SF.multiply(symbolic_circuit, symbolic_circuit)

        # compute the (symbolic) partition function of the squared circuit
        symbolic_partition_function = SF.integrate(symbolic_circuit)

        # instantiate the circuit
        circuit = ctx.compile(symbolic_circuit).to(self.device)
        partition_function = ctx.compile(symbolic_partition_function)

        self.circuit = circuit
        self.partition_function = partition_function
        self.param_dict = self.circuit_params
        self.ctx = ctx
        self.log_Z = self.partition_function().real.flatten()
        self.Z = torch.exp(self.partition_function().real.flatten())

        # normalize the weights -> w_i = (w_iZ_i)/Z, these weights are associated with the
        # normalized input distributions
        weights = self.circuit.layers[-1].weight().flatten()
        part = torch.exp(torch.sum(self.circuit.layers[0].log_partition(), dim=0)) # partition function per component
        weights_n = (weights * part)/self.Z
        self.weights_n = weights_n
        assert torch.isclose(torch.sum(self.weights_n),
                             torch.tensor(1.0, dtype=self.weights_n.dtype,
                                          device=self.weights_n.device))

        # extract positive weights and components for sampling and renormalize the positive weights in isolation
        weights_pos = torch.where(weights > 0, weights, 0)
        self.Z_pos = torch.sum(weights_pos * part, dim=0)
        self.weights_pos_n = (weights_pos * part) / self.Z_pos
        assert torch.isclose(torch.sum(self.weights_pos_n), torch.tensor(1.0, dtype=self.weights_pos_n.dtype, device=self.weights_pos_n.device))

        means = self.circuit.layers[0].mean()
        vars = (self.circuit.layers[0].stddev()) ** 2 # marginal variances per variable and component (D, 4)
        means_pos = torch.stack([means[:, 0], means[:, -1]]) # first and last component are associated with pos. weights
        cov_1 = torch.eye(self.D, device=self.device) * vars[0, 0]
        cov_2 = torch.eye(self.D, device=self.device) * vars[0, -1] # (all rows of vars are identical)
        covs_pos = torch.stack([cov_1, cov_2])  # (2, D, D)
        self.weights_pos_n = torch.stack([self.weights_pos_n[0], self.weights_pos_n[-1]])

        # proposal for rejection sampling
        mix = Categorical(probs=self.weights_pos_n)
        comp = MultivariateNormal(means_pos, covariance_matrix=covs_pos)
        self.q_pos = MixtureSameFamily(mix, comp)

    def log_p(self, x):
        log_p = self.circuit(x).real
        if torch.any(torch.isinf(log_p)):
            print("Warning: clamping -inf log density to -1000 for target")
            log_p = torch.clamp(log_p, min=-1000)
        log_p = log_p[:, 0, 0]
        return log_p - self.log_Z

    def sample(self, num_samples: int):
        ### rejection sampling
        M = self.Z_pos / self.Z
        N_effective = 0
        all_samples = []
        # might give more samples at some points
        while N_effective < num_samples:
            proposed_samples = self.q_pos.sample((num_samples - N_effective,))
            u = torch.distributions.Uniform(0, 1).sample((proposed_samples.shape[0],)).to(self.device)
            # q_{SMM}(x)/M(q_pos(x))
            crit = torch.exp(self.log_p(proposed_samples)) / (M * torch.exp(self.q_pos.log_prob(proposed_samples)))  # log_q_pos is normalized
            accepted = proposed_samples[(u <= crit), :]
            all_samples.append(accepted)
            N_effective += accepted.shape[0]
        samples = torch.vstack(all_samples)
        assert samples.shape[0] == num_samples
        return samples

    @property
    def cov(self):
        # E[XX^T] = \sum_i w_i E_{p_i}[XX^T]
        stddevs = self.circuit.layers[0].stddev()
        assert torch.allclose(stddevs[0,:], stddevs[1,:]) # base distribution is spherical (all variables share components with the same marginal variances)
        assert torch.allclose(self.mean, torch.zeros(self.D).to(self.device))
        return (self.weights_n[0] * torch.eye(self.D, device=self.device)*stddevs[0,0]**2 +
                self.weights_n[1] * torch.eye(self.D, device=self.device)*stddevs[0,1]**2 +
                self.weights_n[2] * torch.eye(self.D, device=self.device)*stddevs[0,2]**2 +
                self.weights_n[3] * torch.eye(self.D, device=self.device)*stddevs[0,3]**2
                )

    @property
    def corr(self):
        cov = self.cov
        std = torch.sqrt(torch.diag(cov))
        corr = cov / torch.outer(std, std)
        return corr

    @property
    def mean(self):
        mean = self.circuit.layers[0].mean()
        return (self.weights_n[0] * mean[:, 0] +
                self.weights_n[1] * mean[:, 1] +
                self.weights_n[2] * mean[:, 2] +
                self.weights_n[3] * mean[:, 3]
                )

class Ellipse(Ring):
    """
    Build an elliptical target from the spherical base distribution.
    """

    def __init__(self, device, base_params, D, mu, M):
        super(Ellipse, self).__init__(device=device, circuit_params=base_params, D=D)

        self.is_even = True
        self.is_elliptic = True
        self.is_log_concave = False
        self.support_cond = True

        self.D = D
        self.M = M.to(self.device)
        self.mu = mu.to(self.device)
        self.M_sqrt = torch.linalg.cholesky(self.M)
        self.M_sqrt_inv = torch.linalg.inv(self.M_sqrt)
        sign, self.log_M_sqrt_inv_det = torch.linalg.slogdet(self.M_sqrt_inv)

        assert sign == 1
        self.p_base = Ring(device=device, circuit_params=base_params, D=D)
        self.device = device
        self.circuit_params = base_params
        self.build_circuit()

    def log_p(self, x):
        x_t = torch.linalg.solve(self.M_sqrt, (x-self.mu).T).T # transform the input
        log_p = self.circuit(x_t).real
        if torch.any(torch.isinf(log_p)):
            print("Warning: clamping -inf log density to -1000 for target")
            log_p = torch.clamp(log_p, min=-1000)
        log_p = log_p[:, 0, 0] - self.log_Z
        return log_p + self.log_M_sqrt_inv_det # add determinant


    def sample(self, num_samples: int):
        samples = self.p_base.sample(num_samples) # sample from base dist.
        assert samples.shape[0] == num_samples
        samples = (self.M_sqrt@samples.T).T + self.mean # transform samples
        return samples

    @property
    def corr(self):
        # abuse of naming M is not the cov but the normalized cov
        cov = self.M
        std = torch.sqrt(torch.diag(cov))
        corr = cov / torch.outer(std, std)
        return corr

    @property
    def cov(self):
        base_cov = self.p_base.cov
        return self.M_sqrt@base_cov@self.M_sqrt.T

    @property
    def mean(self):
        assert torch.allclose(self.p_base.mean, torch.zeros(self.D, device=self.device))
        return self.mu


class Ring2D(Ring):
    def __init__(self, device: str):
        """
        Used for Fig. 2 to illustrate spherical symmetry.
        """
        D = 2
        circuit_params = {
                 "mean": torch.tensor([[0, 0], [0, 0]]),
                 "stddev": torch.sqrt(torch.tensor([[1, 0.3], [1, 0.3]])),
                 "weights": torch.tensor([1 - 1 / D**2, -1 / D**2])
             }
        super().__init__(device=device, circuit_params=circuit_params, D=D)

class Ellipse2D(Ellipse):
    def __init__(self, device: str):
        D = 2
        base_params, mu, M = set_ellipse_params(D)
        super().__init__(device=device, base_params=base_params, D=D, M=M, mu=mu)

class Ellipse4D(Ellipse):
    def __init__(self, device: str):
        D = 4
        base_params, mu, M = set_ellipse_params(D)
        super().__init__(device=device, base_params=base_params, D=D, M=M, mu=mu)

class Ellipse8D(Ellipse):
    def __init__(self, device: str):
        D = 8
        base_params, mu, M = set_ellipse_params(D)
        super().__init__(device=device, base_params=base_params, D=D, M=M, mu=mu)

class Ellipse16D(Ellipse):
    def __init__(self, device: str):
        D = 16
        base_params, mu, M = set_ellipse_params(D)
        super().__init__(device=device, base_params=base_params, D=D, M=M, mu=mu)
