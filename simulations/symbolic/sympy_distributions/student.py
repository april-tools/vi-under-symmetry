import sympy as sp

def student_sympy(eta, loc, scale, df):
    z = (eta - loc) / scale
    return (sp.gamma((df + 1) / 2) /(sp.sqrt(df * sp.pi) * sp.gamma(df / 2) * scale)) * (1 + z ** 2 / df) ** (-(df + 1) / 2)

def log_student_sympy(eta, loc, scale, df):
    z = (eta - loc) / scale
    half = sp.Rational(1, 2)

    return (
        sp.log(sp.gamma((df + 1) * half))
        - sp.log(sp.gamma(df * half))
        - half * sp.log(df * sp.pi)
        - sp.log(scale)
        - ((df + 1) * half) * sp.log(1 + z**2 / df)
    )