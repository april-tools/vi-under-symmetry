"""
Plot a given density.
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_density(density,
                 lower,
                 upper,
                 plt_title,
                 norm=None,
                 steps=500,
                 reference_mean=None):
    with torch.no_grad():
        x = torch.linspace(lower, upper, steps).to(density.device)
        x = torch.stack(torch.meshgrid(x, x), dim=-1).reshape(-1, 2)

        y = density.log_p(x)
        y = torch.exp(y)
        y = y.detach()

        x = x.cpu().numpy()
        y = y.cpu().numpy()

        if norm is None:  # if no norm is passed, define it based on max value
            norm = mcolors.Normalize(
                vmin=0.0,
                vmax=1.1 * y.max(),
            )

        single_fig, single_ax = plt.subplots(figsize=(8, 6))
        plt.contourf(
        x[:, 0].reshape(steps, steps),
        x[:, 1].reshape(steps, steps),
        y.reshape(steps, steps),
        levels=15,
        cmap="viridis",
        norm=norm
        )
        alt_mean = None
        if reference_mean is None:
            mean = density.mean
        else:
            mean = reference_mean
            alt_mean = density.mean
        if alt_mean is not None:
            plt.scatter(alt_mean[0].cpu(), alt_mean[1].cpu(), marker='s', color="yellow", s=500)
        plt.scatter(mean[0].cpu(), mean[1].cpu(), marker='*', color="crimson", s=600)
        single_ax.xaxis.set_tick_params(labelbottom=False)
        single_ax.yaxis.set_tick_params(labelleft=False)
        single_ax.set_xticks([])
        single_ax.set_yticks([])
        plt.xlim(lower, upper)
        plt.ylim(lower, upper)
        plt.savefig(plt_title, bbox_inches="tight")
        plt.close()
        return norm
