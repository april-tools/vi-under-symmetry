"""
Reproduces Fig. 7 in the appendix.
"""
import os
import torch
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import scipy
from matplotlib.colors import Normalize
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

plot_folder = os.path.join("figures/appendix_figure_7")
os.makedirs(plot_folder, exist_ok=True)

levels = 8
upper = 5.5
lower = -upper

eta_1, eta_2 = sp.symbols('eta_1 eta_2')
eta = sp.Matrix([eta_1, eta_2])

# this is even symmetric but not elliptical symmetric
mu_total_1 = 2
mu_total_2 = 1
mu_total = sp.Matrix([mu_total_1, mu_total_2])
mu_total_np = np.array([mu_total_1, mu_total_2])

# place the components
mode_separation = sp.Matrix([1.5, 1.5])
mu1 = mu_total + mode_separation
mu2 = mu_total - mode_separation

cov_p = 0.7

scale_p = sp.Matrix([
    [1, cov_p],
    [cov_p, 2]
])

scale_w = sp.Matrix([[1., 0], [0, 1.]]) # radial

def log_gaussian_sp(eta, loc, scale_mat):
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

# individual components
p1 = sp.exp(log_gaussian_sp(eta, mu1, scale_mat=scale_p))
p2 = sp.exp(log_gaussian_sp(eta, mu2, scale_mat=scale_p))

# Mixture (equivalent to total_p)
total_p = sp.simplify(0.5 * p1 + 0.5 * p2)
total_p_func = sp.lambdify((eta_1, eta_2), total_p,  "numpy")

# Create grid
x = torch.linspace(lower, upper, 200)
y = torch.linspace(lower, upper, 200)
X, Y = torch.meshgrid(x, y, indexing="ij")

# Stack grid into positions
pos = torch.stack([X, Y], dim=-1).numpy()

# Evaluate PDF
V = total_p_func(pos[..., 0], pos[..., 1])


# mu w
w_mu = sp.simplify(-log_gaussian_sp(eta=eta, loc=mu_total, scale_mat=scale_w))
w_mu_func = sp.lambdify((eta_1, eta_2), w_mu, "numpy")

# centered w
w_c = sp.simplify(-log_gaussian_sp(eta=eta, loc=sp.Matrix([0., 0.]), scale_mat=scale_w))
w_c_func = sp.lambdify((eta_1, eta_2), w_c, "numpy")

# centered p
p1_c = sp.exp(log_gaussian_sp(eta + mu_total, mu1, scale_p))
p2_c = sp.exp(log_gaussian_sp(eta + mu_total, mu2, scale_p))
total_p_c = sp.simplify(0.5 * p1_c + 0.5 * p2_c)
total_p_c_func = sp.lambdify((eta_1, eta_2), total_p_c, "numpy")
p_vals_c = total_p_c_func(pos[..., 0], pos[..., 1])

int_symb = sp.simplify(w_c * total_p_c)
f_num = sp.lambdify((eta_1, eta_2), int_symb, modules="numpy")
I_base, err_base = scipy.integrate.nquad(
            f_num,
            [[-100, 100], [-100, 100]]
        )
print("Min KL", I_base, err_base)

shift_size = [-1.8]
shift_radius = [3.3] #np.linspace(0, 8, 9)
print("shift radius", shift_radius)

# reference vals
p_vals = total_p_c_func(pos[..., 0], pos[..., 1])
w_mu_vals = w_c_func(pos[..., 0], pos[..., 1])

norm = Normalize(vmin=np.min(p_vals).item() * 0.9, vmax=np.max(p_vals).item() * 1.1)

plt.contourf(X.numpy(), Y.numpy(), V, norm=norm, levels=levels)
plt.savefig(os.path.join(plot_folder, "target.pdf"), bbox_inches='tight')
plt.close()

for s in shift_size:
    print("\n\n")
    print("Shift", s)
    for r_scale in shift_radius:
        r = r_scale * np.pi/4
        v_prime = sp.Matrix([sp.cos(r)*s, sp.sin(r)*s])
        v = np.array([np.cos(r)*s, np.sin(r)*s])

        eta_1, eta_2 = sp.symbols('eta_1 eta_2')
        eta = sp.Matrix([eta_1, eta_2])
        w_v_prime = -log_gaussian_sp(eta=eta, loc=v_prime, scale_mat=scale_w) #MultivariateNormal(torch.zeros(2)+v_prime, torch.tensor([[1., 0.], [0., 1.]]))
        w_v_func = sp.lambdify((eta_1, eta_2), w_v_prime, "numpy")
        int_symb = sp.simplify(w_v_prime * total_p_c)
        f_num = sp.lambdify((eta_1, eta_2), int_symb, modules="numpy")
        I, err = scipy.integrate.nquad(
            f_num,
            [[-100, 100], [-100, 100]]
        )

        print(f"v: ({v_prime[0]}, {v_prime[1]}), \n"
              f"radius: {r},  \n"
              f"KL: {I} ({err}), \n"
              f"KL-KL(best): {I-I_base}, \n\n")

        J = (w_v_func(pos[..., 0], pos[..., 1]) - w_mu_vals)
        W_pos = np.ma.masked_where(J < 0, p_vals)
        W_neg = np.ma.masked_where(J >= 0, p_vals)

        # red where weight difference < 0
        plt.contourf(
            X.numpy(), Y.numpy(), W_neg,
            levels=8,
            cmap="Reds",
            norm=norm
        )

        # blue where weight difference > 0
        plt.contourf(
            X.numpy(), Y.numpy(), W_pos,
            levels=8,
            cmap="Blues",
            norm=norm
        )

        plt.contour(
            X.numpy(), Y.numpy(), J,
            levels=[0],
            linewidths=3,
        )

        # partitioned hyperplane
        c = -np.dot(v, v) / 2 # -||v||^2
        Z = v[0] * X + v[1] * Y # <v, x>
        plt.text(-4, 4, r"$\mathcal{H}_1$", fontsize=30)
        plt.text(4, -4, r"$\mathcal{H}_2$", fontsize=30)
        plt.savefig(os.path.join(plot_folder, f"domain_partitioning_shift_{s}_r_{r_scale}_H_1_H_2.pdf"), bbox_inches='tight')
        plt.close()

        plt.contourf(
            X.numpy(), Y.numpy(), W_neg,
            levels=8,
            cmap="Reds",
            norm=norm
        )

        # blue where weight difference > 0
        plt.contourf(
            X.numpy(), Y.numpy(), W_pos,
            levels=8,
            cmap="Blues",
            norm=norm
        )

        plt.contour(
            X.numpy(), Y.numpy(), J,
            levels=[0],
            linewidths=3,
        )

        # partitioned hyperplane
        c = -np.dot(v, v) / 2  # -||v||^2
        Z = v[0] * X + v[1] * Y  # <v, x>
        plt.text(-4, 4, r"$\mathcal{H}_3$", fontsize=30)
        plt.text(-0.3, 0, r"$\mathcal{H}_4$", fontsize=30)
        plt.text(4, -4, r"$\mathcal{H}_2$", fontsize=30)
        plt.contour(X, Y, Z, levels=[c], linewidths=3, linestyles="dashed")
        plt.savefig(os.path.join(plot_folder, f"domain_partitioning_shift_{s}_r_{r_scale}__H_2_H_3_H_4.pdf"),
                    bbox_inches='tight')
        plt.close()