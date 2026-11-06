import sympy as sp

def gaussian_sympy(eta, loc: float, scale: float, eps=0):
    # print(f"Safety value of {eps} used.")
    assert scale > 0
    g = (1 / (sp.sqrt(2 * sp.pi) * scale)) * sp.exp(-(eta - loc) ** 2 / (2 * scale ** 2))
    return g + eps

def log_gaussian_sympy(eta, loc: float, scale: float):
    assert scale > 0
    return (
            -sp.log(sp.sqrt(2 * sp.pi) * scale)
            - (eta - loc) ** 2 / (2 * scale ** 2)
    )

def log_gaussian_sympy_HD(eta, loc, scale_mat):
    n = eta.shape[0]

    scale_inv = scale_mat.inv()
    det_scale = scale_mat.det()

    diff = eta - loc
    quad_form = (diff.T * scale_inv * diff)[0]

    return sp.simplify(
        -sp.Rational(1, 2) * (
            n * sp.log(2 * sp.pi)
            + sp.log(det_scale)
            + quad_form
        )
    )

def mog_sympy(eta, loc, scale):
    scale_sp = sp.Float(scale)
    loc_sp = sp.Float(loc)
    mog = (sp.Float(0.5) * (1 / (sp.sqrt(2 * sp.pi) * scale_sp)) * sp.exp(-(eta - loc_sp) ** 2 / (2 * scale_sp ** 2)) +
           sp.Float(0.5) * (1 / (sp.sqrt(2 * sp.pi) * scale_sp)) * sp.exp(-(eta + loc_sp) ** 2 / (2 * scale_sp ** 2)))
    return mog
    #return (1/2) * density(Normal(eta, loc, scale))(eta) + (1/2) * density(Normal(eta, -loc, scale))(eta)
