import sympy as sp
import os
from simulations.symbolic.targets import Target2
from simulations.symbolic.sympy_distributions.student import log_student_sympy

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
n = int(round(upper / steps))  # number of steps from 0 to upper
m_range = np.arange(-n, n + 1) * steps

alphas = [0.5, 2.0]
scale_q = 4
df = 5

eta = sp.symbols('eta', real=True)

plot_folder = f"figures/mean_recovery/theorem_4_case_1"
os.makedirs(plot_folder, exist_ok=True)
plot_folder_2 = f"figures/mean_recovery/theorem_4_case_2"
os.makedirs(plot_folder_2, exist_ok=True)

p = Target2()

for alpha in alphas:
    div_vals = []
    print(f"Starting alpha={alpha}")
    v = sp.symbols('v', real=True)

    def interval_integral(a, b, v, alpha):
        expr = sp.exp(((1-alpha)*log_student_sympy(eta, loc=v, scale=scale_q, df=df)))
        return sp.integrate(expr, (eta, a, b), meijerg=False)

    print(f"Symbolically integrating ....")
    # integrate over each component
    div_expr = ((p.p_val_1 ** alpha) * interval_integral(p.l_1, p.u_1, v, alpha=alpha)
                + (p.p_val_2 ** alpha) * interval_integral(p.l_2, p.u_2, v, alpha=alpha)
                + (p.p_val_m ** alpha) * interval_integral(p.l_m, p.u_m, v, alpha=alpha))

    for m in m_range:
        val = 1 / (alpha*(alpha-1)) * div_expr.subs(v, m)
        val = val.evalf(64) - 1/(alpha*(alpha-1))  # faster than val.evalf()
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
    if alpha >= 1 + 2/(df+1):
        plt.savefig(f"{plot_folder}/div_curve_{alpha}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{plot_folder_2}/div_curve_{alpha}.pdf", bbox_inches='tight')
    plt.close()





