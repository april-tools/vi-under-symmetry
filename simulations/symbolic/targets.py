"""
Targets used for symbolic experiments.
"""

import sympy as sp

def uniform_density_sympy(eta, l, u):
    assert l < u
    return sp.Piecewise(
        (1 / (u - l), (eta >= l) & (eta <= u)),
        (0, True)
    )


class Target1:
    def __init__(self):
        self.w_1 = 0.5
        self.w_2 = 1-self.w_1

        self.l_1 = -9
        self.u_1 = -3

        ### density of the mixture on the first interval
        self.p_val_1 =  self.w_1 * 1/(self.u_1 - self.l_1)

        self.l_2 = 3
        self.u_2 = 9

        ### density of the mixture on the first interval
        self.p_val_2 = self.w_2 * 1 / (self.u_2 - self.l_2)

    def p_sympy(self, eta):
        p_1 = uniform_density_sympy(eta, self.l_1, self.u_1)
        p_2 = uniform_density_sympy(eta, self.l_2, self.u_2)
        return self.w_1*p_1 + self.w_2*p_2

    def log_p_sympy(self, eta):
        pdf = self.p_sympy(eta)
        return sp.Piecewise(
            (sp.log(pdf), pdf > 0),
            (-sp.oo, True)
        )



class Target2:
    def __init__(self):
        """
        Mix Target1 with another uniform in the center.
        """
        self.w_m = 0.01
        self.w_1 = 0.5 * (1-self.w_m)
        self.w_2 = 0.5 * (1-self.w_m)

        assert self.w_m + self.w_1 + self.w_2 == 1.0

        self.l_1 = -9
        self.u_1 = -3

        ### density of the mixture on the first interval
        self.p_val_1 = self.w_1 * 1 / (self.u_1 - self.l_1)

        self.l_2 = 3
        self.u_2 = 9

        ### density of the mixture on the second interval
        self.p_val_2 = self.w_2 * 1 / (self.u_2 - self.l_2)

        self.l_m = -0.3
        self.u_m = 0.3

        ### density of the mixture on the middle interval
        self.p_val_m = self.w_m * 1 / (self.u_m - self.l_m)

    def p_sympy(self, eta):
        p_1 = uniform_density_sympy(eta, self.l_1, self.u_1)
        p_2 = uniform_density_sympy(eta, self.l_2, self.u_2)
        p_m = uniform_density_sympy(eta, self.l_m, self.u_m)
        return self.w_1 * p_1 + self.w_2 * p_2 + self.w_m * p_m

    def log_p_sympy(self, eta):
        pdf = self.p_sympy(eta)
        return sp.Piecewise(
            (sp.log(pdf), pdf > 0),
            (-sp.oo, True)
        )

