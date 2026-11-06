from simulations.symbolic.sympy_distributions.laplace import log_laplace_sympy
from simulations.symbolic.sympy_distributions.gaussian import log_gaussian_sympy
from simulations.symbolic.targets import Target1
import sympy as sp
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams["axes.labelsize"] = 40
plt.rcParams['axes.linewidth'] = 4

star_size = 30
square_size = 20
accent_color_good = "green"
accent_color_bad = "red"
accent_alpha = 0.7
linewidth = 5
steps = 0.01

loc_p = 6
scale_p =  3
scale_q =  4

upper = 15

eta = sp.symbols('eta')
p = Target1()
p_pdf_sympy = p.p_sympy(eta)
p_logpdf_sympy = p.log_p_sympy(eta)

plot_folder = f"figures/mean_recovery/theorem_1_case_1"
plot_folder_2 = f"figures/mean_recovery/theorem_1_case_2"
os.makedirs(plot_folder, exist_ok=True)
os.makedirs(plot_folder_2, exist_ok=True)

print("Computing true mean...")
mu_p = sp.integrate(eta*p_pdf_sympy, (eta, -sp.oo, sp.oo))
print("True mean:", mu_p)

n = int(round(upper / steps))
m_range = np.arange(-n, n + 1) * steps

p_vals = np.array([p_pdf_sympy.subs(eta, x) for x in m_range], dtype=float)

plt.plot(m_range, p_vals, linewidth=linewidth)
plt.xlabel(r"$x$")
plt.xticks([])
plt.yticks([])
plt.margins(y=0.1)
plt.savefig(f"{plot_folder}/problem.pdf", bbox_inches='tight')
plt.savefig(f"{plot_folder_2}/problem.pdf", bbox_inches='tight')
plt.close()

### laplace
v = sp.symbols('v')
log_q = log_laplace_sympy(eta=eta, loc=v, scale=scale_q)
log_q_0 = log_laplace_sympy(eta=eta, loc=0, scale=1) # base distribution

# integrands on the twp intervals
integrand_1 = (sp.log(p.p_val_1) - log_q) * p.p_val_1
integrand_2 = (sp.log(p.p_val_2) - log_q) * p.p_val_2

print(f"Symbolically integrating ....")
kl_expr = (sp.integrate(integrand_1, (eta, p.l_1, p.u_1), meijerg=False)
           + sp.integrate(integrand_2, (eta, p.l_2, p.u_2), meijerg=False)).evalf(50)
print("KL formula:", kl_expr)

w_vals = [-log_q_0.subs(eta, x) for x in m_range]

kl_vals = []
for m in m_range:
    val = kl_expr.subs(v, m)
    val = val.evalf()  # faster than val.evalf()
    print(f"KL (v={m}):", val)
    kl_vals.append(val)

min_kl_val = np.min(np.array(kl_vals))
chosen_mean = m_range[np.where(np.array(kl_vals) == min_kl_val)[0]]
mask = np.array(kl_vals) == min_kl_val
print("Means:", chosen_mean)
print("Number of means:", len(chosen_mean))

plt.plot(m_range, kl_vals, linewidth=linewidth)
plt.xlabel(r"$\nu$")
if len(chosen_mean) == 1:
    plt.plot(chosen_mean, min_kl_val, marker='*',  color=accent_color_good, markersize=star_size, alpha=accent_alpha)
else:
    plt.fill_between(
        m_range[mask],
        float(min_kl_val) - 0.1 * float(min_kl_val),
        float(min_kl_val) + 0.1 * float(min_kl_val),
        color=accent_color_bad,
        alpha=0.7
    )
plt.xticks([])
plt.yticks([])
plt.margins(y=0.1)
plt.savefig(f"{plot_folder_2}/div_curve.pdf", bbox_inches='tight')
plt.close()

plt.plot(m_range, w_vals, linewidth=linewidth)
plt.xlabel(r"$x$")
plt.xticks([])
plt.yticks([])
plt.margins(y=0.1)
plt.savefig(f"{plot_folder_2}/w_curve.pdf", bbox_inches='tight')
plt.close()


### Gaussian
v = sp.symbols('v')
log_q = log_gaussian_sympy(eta=eta, loc=v, scale=scale_q)
log_q_0 = log_gaussian_sympy(eta=eta, loc=0, scale=1)

# integrands on the twp intervals
integrand_1 = (sp.log(p.p_val_1) - log_q) * p.p_val_1
integrand_2 = (sp.log(p.p_val_2) - log_q) * p.p_val_2

print(f"Symbolically integrating ....")
kl_expr = (sp.integrate(integrand_1, (eta, p.l_1, p.u_1), meijerg=False)
           + sp.integrate(integrand_2, (eta, p.l_2, p.u_2), meijerg=False)).evalf(50)
print("KL formula:", kl_expr)

n = int(round(upper / steps))  # number of steps from 0 to upper
m_range = np.arange(-n, n + 1) * steps
kl_vals = []

w_vals = [-log_q_0.subs(eta, x) for x in m_range]

for m in m_range:
    val = kl_expr.subs(v, m)
    val = val.evalf()
    print(f"KL (v={m}):", val)
    kl_vals.append(val)

min_kl_val = np.min(np.array(kl_vals))
chosen_mean = m_range[np.where(np.array(kl_vals) == min_kl_val)[0]]
mask = np.array(kl_vals) == min_kl_val
print("Means:", chosen_mean)
print("Number of means:", len(chosen_mean))

plt.plot(m_range, kl_vals, linewidth=linewidth)
plt.xlabel(r"$\nu$")
if len(chosen_mean) == 1:
    plt.plot(chosen_mean, min_kl_val, marker='*', color=accent_color_good, markersize=star_size, alpha=accent_alpha)
else:
    plt.fill_between(
        m_range[mask],
        float(min_kl_val) - 0.1 * float(min_kl_val),
        float(min_kl_val) + 0.1 * float(min_kl_val),
        color=accent_color_bad,
        alpha=accent_alpha
    )
plt.xticks([])
plt.yticks([])
plt.margins(y=0.1)
plt.savefig(f"{plot_folder}/div_curve.pdf", bbox_inches='tight')
plt.close()

plt.plot(m_range, w_vals, linewidth=linewidth)
plt.xlabel(r"$x$")
plt.xticks([])
plt.yticks([])
plt.margins(y=0.1)
plt.savefig(f"{plot_folder}/w_curve.pdf", bbox_inches='tight')
plt.close()





