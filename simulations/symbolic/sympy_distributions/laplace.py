import sympy as sp

def log_laplace_sympy(eta, loc, scale):
    assert scale > 0
    return (
        -sp.log(2 * scale)
        - sp.sqrt((eta - loc) ** 2) / scale
    )

def log_laplace_sympy_abs(eta, loc, scale):
    assert scale > 0
    return (
        -sp.log(2 * scale)
        - sp.Abs((eta - loc)) / scale
    )