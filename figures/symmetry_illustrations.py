"""
Reproduces the symmetry illustrations in Figure 2.
"""

from simulations.stochastic.targets import Ring2D, GMMTarget2DFig, AGMMTarget2DFig
from matplotlib.patches import Ellipse
import numpy as np
import os
import torch
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt_folder = "figures/symmetry_illustrations"
os.makedirs(plt_folder, exist_ok=True)
ring = Ring2D("cpu")
mean = ring.mean

lower = -2.5
upper = 2.5
steps = 100
levels = 10
x = torch.linspace(lower, upper, steps)
x = torch.stack(torch.meshgrid(x, x), dim=-1).reshape(-1, 2)
y = ring.log_p(x)
y = torch.exp(y)
y = y.detach()
x = x.cpu().numpy()
y = y.cpu().numpy()

norm = mcolors.Normalize(
        vmin=0.0,
        vmax=1 * y.max(),
)

single_fig, single_ax = plt.subplots(figsize=(8, 6))
plt.contourf(
        x[:, 0].reshape(steps, steps),
        x[:, 1].reshape(steps, steps),
        y.reshape(steps, steps),
        levels=10,
        cmap="viridis",
        norm=norm
)
plt.scatter(mean[0].cpu(), mean[1].cpu(), marker='*', color="crimson", s=600)
ellipse = Ellipse(xy=(mean.numpy()), width=ring.cov[0, 0]*2, height=ring.cov[0, 0]*2,
                        edgecolor='magenta', fc='None', lw=5)
single_ax.add_patch(ellipse)
single_ax.set_aspect("equal")
single_ax.xaxis.set_tick_params(labelbottom=False)
single_ax.yaxis.set_tick_params(labelleft=False)
single_ax.set_xticks([])
single_ax.set_yticks([])
plt.xlim(lower, upper)
plt.ylim(lower, upper)
plt.savefig(os.path.join(plt_folder, "spherical.pdf"), bbox_inches="tight")
#plt.show()
plt.close()

# transform into an ellipse
M = torch.ones(2, 2)*0.5
M.fill_diagonal_(1.0)
sqrt_M = torch.linalg.cholesky(M)
sqrt_M_inv = torch.linalg.inv(sqrt_M)

# https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
vals, vecs = np.linalg.eigh(M)
width  = np.sqrt(vals[0])
height = np.sqrt(vals[1])
angle  = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
ellipse = Ellipse((0, 0), width=width*2, height=height*2, angle=angle, edgecolor="magenta", fc='None', lw=5)

x_t = (sqrt_M_inv@x.T).T
y_t = ring.log_p(x_t)
y_t = torch.exp(y_t) * torch.linalg.det(sqrt_M_inv)

single_fig, single_ax = plt.subplots(figsize=(8, 6))
plt.contourf(
        x[:, 0].reshape(steps, steps),
        x[:, 1].reshape(steps, steps),
        y_t.reshape(steps, steps),
        levels=10,
        cmap="viridis",
        norm=norm
)
single_ax.add_patch(ellipse)
plt.scatter(mean[0].cpu(), mean[1].cpu(), marker='*', color="crimson", s=600)
single_ax.set_aspect("equal")
single_ax.xaxis.set_tick_params(labelbottom=False)
single_ax.yaxis.set_tick_params(labelleft=False)
single_ax.set_xticks([])
single_ax.set_yticks([])
plt.xlim(lower, upper)
plt.ylim(lower, upper)
plt.savefig(os.path.join(plt_folder, "elliptical.pdf"), bbox_inches="tight")
#plt.show()
plt.close()

### symmetric gmm
lower = -6.5
upper = 6.5
steps = 100

gmm = GMMTarget2DFig(device="cpu")
mean = gmm.mean
x = torch.linspace(lower, upper, steps)
x = torch.stack(torch.meshgrid(x, x), dim=-1).reshape(-1, 2)
y = gmm.log_p(x)
y = torch.exp(y)
y = y.detach()
x = x.cpu().numpy()
y = y.cpu().numpy()

norm = mcolors.Normalize(
        vmin=0.0,
        vmax=1 * y.max(),
)

single_fig, single_ax = plt.subplots(figsize=(8, 6))
plt.contourf(
        x[:, 0].reshape(steps, steps),
        x[:, 1].reshape(steps, steps),
        y.reshape(steps, steps),
        levels=6,
        cmap="viridis",
        norm=norm
)
y_line = -x
x_1 = torch.tensor([-2.7, -2.7])
x_2 = -x_1
#plt.plot(x, y_line, linestyle='--', linewidth=3, color="orange", alpha=0.9, zorder=1)
plt.scatter(mean[0].cpu(), mean[1].cpu(), marker='*', color="crimson", s=600, zorder=3)
plt.scatter(x_1[0].cpu(), x_1[1].cpu(), color="magenta", s=200, zorder=3)
plt.scatter(x_2[0].cpu(), x_2[1].cpu(), color="magenta", s=200, zorder=3)
single_ax.set_aspect("equal")
single_ax.xaxis.set_tick_params(labelbottom=False)
single_ax.yaxis.set_tick_params(labelleft=False)
single_ax.set_xticks([])
single_ax.set_yticks([])
plt.xlim(lower, upper)
plt.ylim(lower, upper)
plt.savefig(os.path.join(plt_folder, "gmm.pdf"), bbox_inches="tight")
#plt.show()
plt.close()

agmm = AGMMTarget2DFig(device="cpu")
x = torch.linspace(lower, upper, steps)
x = torch.stack(torch.meshgrid(x, x), dim=-1).reshape(-1, 2)
y = agmm.log_p(x)
y = torch.exp(y)
y = y.detach()
x = x.cpu().numpy()
y = y.cpu().numpy()

single_fig, single_ax = plt.subplots(figsize=(8, 6))
plt.contourf(
        x[:, 0].reshape(steps, steps),
        x[:, 1].reshape(steps, steps),
        y.reshape(steps, steps),
        levels=6,
        cmap="viridis",
        norm=norm
)
single_ax.set_aspect("equal")
single_ax.xaxis.set_tick_params(labelbottom=False)
single_ax.yaxis.set_tick_params(labelleft=False)
single_ax.set_xticks([])
single_ax.set_yticks([])
plt.xlim(lower, upper)
plt.ylim(lower, upper)
plt.savefig(os.path.join(plt_folder, "agmm.pdf"), bbox_inches="tight")
plt.close()



