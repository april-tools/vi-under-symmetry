import torch

def fkl(p, q, S: int):
    """
    Compute FKL from target samples.
    """
    samples = p.sample(S)
    return torch.mean(p.log_p(samples) - q.log_p(samples))