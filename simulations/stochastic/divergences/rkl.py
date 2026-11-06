import torch

def rkl_reparam(p, q, S: int):
    """
    RKL with reparameterization.
    """
    samples = q.sample(S)
    return torch.mean(q.log_p(samples) - p.log_p(samples))


def rkl_reinforce(p, q, S: int):
    """
    RKL with REINFORCE.
    """
    samples = q.sample(S).detach()
    w = (q.log_p(samples) - p.log_p(samples)).detach()
    return torch.mean(w*q.log_p(samples))


def rkl_value(p, q, S: int):
    """
    Estimate RKL value, no gradients.
    """
    with torch.no_grad():
        samples = q.sample(S)
        return torch.mean(q.log_p(samples) - p.log_p(samples))
