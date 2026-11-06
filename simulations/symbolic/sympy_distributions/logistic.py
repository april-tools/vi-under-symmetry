import sympy as sp

def log_logistic_sympy(eta, loc, scale):
    assert scale > 0
    z = (eta - loc) / scale
    return (
        -sp.log(scale)
        - z
        - 2 * sp.log(1 + sp.exp(-z))
    )