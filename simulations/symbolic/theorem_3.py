import sympy as sp
import os
from simulations.symbolic.targets import Target1
from simulations.symbolic.sympy_distributions.gaussian import log_gaussian_sympy, gaussian_sympy

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams["axes.labelsize"] = 45
plt.rcParams['axes.linewidth'] = 4

star_size = 30
square_size = 20
accent_color_good = "green"
accent_color_bad = "red"
linewidth = 5
accent_alpha = 0.7

lower = -15
upper = 15
steps = 0.01
n = int(round(upper / steps))
m_range = np.arange(-n, n + 1) * steps

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
scale_q = 4

eta = sp.symbols('eta', real=True)

plot_folder = f"figures/mean_recovery/theorem_3_case_1"
os.makedirs(plot_folder, exist_ok=True)
plot_folder_2 = f"figures/mean_recovery/theorem_3_case_2"
os.makedirs(plot_folder_2, exist_ok=True)

p = Target1()

for alpha in alphas:
    div_vals = []
    print(f"Starting alpha={alpha}")
    v = sp.symbols('v', real=True)

    def interval_integral(a, b, v, alpha):
        expr = sp.exp(((1-alpha)*log_gaussian_sympy(eta, loc=v, scale=scale_q)))
        return sp.integrate(expr, (eta, a, b), meijerg=False)

    print(f"Symbolically integrating ....")
    div_expr = ((p.p_val_1 ** alpha) * interval_integral(p.l_1, p.u_1, v, alpha)
                + (p.p_val_2 ** alpha) * interval_integral(p.l_2, p.u_2, v, alpha))

    for m in m_range:
        val = 1 / (alpha*(alpha-1)) * div_expr.subs(v, m)
        val = val.evalf(64) - 1/(alpha*(alpha-1))
        print(f"Div (v={m}):", val)
        div_vals.append(val)

    div_vals = np.array(div_vals).astype(float)
    min_div_val = np.min(div_vals).astype(float)
    chosen_mean = m_range[np.where(np.isclose(div_vals, min_div_val, rtol=10 ** (-8), atol=10 ** (-8)))[0]] #m_range[np.where(kl_vals == min_kl_val)[0]]
    mask = np.where(div_vals == min_div_val)
    print("Means:", chosen_mean)
    print("Number of means:", len(chosen_mean))

    plt.plot(m_range, div_vals, linewidth=linewidth)
    plt.xlabel(r"$\nu$")
    if len(chosen_mean) == 1:
        plt.plot(chosen_mean, min_div_val, marker='*', color=accent_color_good, markersize=star_size, alpha=accent_alpha)
    else:
        for c in chosen_mean:
            plt.plot(c, min_div_val, marker='s', color=accent_color_bad, markersize=square_size, alpha=accent_alpha)

    plt.margins(y=0.1)
    plt.xticks([])
    plt.yticks([])
    if alpha > 1:
        plt.savefig(f"{plot_folder}/div_curve_{alpha}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{plot_folder_2}/div_curve_{alpha}.pdf", bbox_inches='tight')
    plt.close()

    q_0 = gaussian_sympy(eta=eta, loc=0, scale=1)  # base distribution
    w_vals = [1/(alpha*(alpha-1))*(q_0.subs(eta, x)**(1-alpha)) for x in m_range]
    plt.plot(m_range, w_vals, linewidth=linewidth)
    plt.xlabel(r"$x$")
    plt.margins(y=0.1)
    plt.xticks([])
    plt.yticks([])
    if alpha > 1:
        plt.savefig(f"{plot_folder}/w_curve_{alpha}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{plot_folder_2}/w_curve_{alpha}.pdf", bbox_inches='tight')
    plt.close()




