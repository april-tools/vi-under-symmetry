"""
Illustrations of weighting curves based on variational families (Fig.5 and Fig. 6)
"""

import matplotlib.pyplot as plt
from torch.distributions import Normal, Laplace, StudentT, Cauchy
from pyro.distributions import Logistic
import torch
import matplotlib
import os
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['xtick.major.width'] = 2  # tick thickness
plt.rcParams['xtick.major.size'] = 8   # tick length
plt.rcParams['axes.linewidth'] = 4
steps = 0.01
linewidth = 5

plot_folder = os.path.join("figures/w_plots")
os.makedirs(plot_folder, exist_ok=True)
x_range = torch.arange(-10, 10+steps, step=steps)
scale = 1

dists = ["gaussian", "laplace", "student", "cauchy", "logistic"]

for d in dists:
    os.makedirs(os.path.join(plot_folder, d), exist_ok=True)
    if d == "gaussian":
        dist = Normal(loc=0, scale=scale)
    elif d == "laplace":
        dist = Laplace(loc=0, scale=scale)
    elif d == "student":
        dist = StudentT(loc=0, scale=scale, df=5)
    elif d == "cauchy":
        dist = Cauchy(loc=0, scale=scale)
    elif d == "logistic":
        dist = Logistic(loc=0, scale=scale)
    else:
        raise ValueError("Distribution misspecified.")

    y = -dist.log_prob(x_range)
    single_fig, single_ax = plt.subplots(figsize=(6, 4))
    plt.plot(x_range, y, linewidth=linewidth)
    single_ax.yaxis.set_tick_params(labelleft=False)
    single_ax.set_yticks([])
    single_ax.set_xticks([0])
    single_ax.set_xticklabels(['0'])
    plt.savefig(os.path.join(plot_folder, f"{d}/wkl.pdf"), bbox_inches="tight")
    plt.close()

    for alpha in [0.1, 0.5, 0.9, 1.1, 2.0]:
        y = 1/(alpha*(alpha-1))*torch.exp(dist.log_prob(x_range))**(1-alpha)
        single_fig, single_ax = plt.subplots(figsize=(6, 4))
        plt.plot(x_range, y, linewidth=linewidth)
        single_ax.yaxis.set_tick_params(labelleft=False)
        single_ax.set_yticks([])
        single_ax.set_xticks([0])
        single_ax.set_xticklabels(['0'])
        plt.savefig(os.path.join(plot_folder, f"{d}/w_alpha_{alpha}.pdf"), bbox_inches="tight")
        plt.close()
