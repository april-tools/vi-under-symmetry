import torch


def alpha_div_reparam(p, q, S: int, alpha: float):
    """
    Alpha divergence with reparameterization.
    """
    assert alpha > 0 and alpha != 1
    samples = q.sample(S)
    log_w = alpha * (p.log_p(samples) - q.log_p(samples))
    log_mean_exp = torch.logsumexp(log_w, dim=0) - torch.log(
        torch.tensor(S, device=log_w.device)
    )

    return (torch.exp(log_mean_exp) - 1.0) / (alpha * (alpha - 1))


def alpha_div_reinforce(p, q, S, alpha: float):
    """
    Alpha divergence with REINFORCE.
    """
    assert alpha > 0 and alpha != 1
    samples = q.sample(S).detach()
    w = torch.exp(alpha * (p.log_p(samples) - q.log_p(samples))).detach() # (p^a/q^a)
    expr = torch.mean(w*(1-alpha)*q.log_p(samples)) # total mean

    return (expr - 1.0)/(alpha*(alpha-1))


def alpha_div_value(p, q, S: int, alpha: float):
    """
    Estimate the alpha-div value, no gradients.
    """
    assert alpha > 0 and alpha != 1
    with torch.no_grad():
        samples = q.sample(S)

        log_p = p.log_p(samples)
        log_q = q.log_p(samples)

        log_w = alpha * (log_p - log_q)

        log_mean_w = torch.logsumexp(log_w, dim=0) - torch.log(
            torch.tensor(S, device=log_w.device)
        )

        mean_w = torch.exp(log_mean_w)

        return ((mean_w - 1.0) / (alpha * (alpha - 1)))


