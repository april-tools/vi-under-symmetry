import argparse
import torch
torch.set_printoptions(precision=12)
import torch.optim as optim
from simulations.stochastic.divergences.alpha import alpha_div_reparam, alpha_div_reinforce, alpha_div_value
from simulations.stochastic.divergences.fkl import fkl
from simulations.stochastic.divergences.rkl import rkl_reparam, rkl_reinforce, rkl_value
from simulations.stochastic.metrics import delta_mean, delta_corr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os
from simulations.stochastic.set_seed import set_seed
from simulations.stochastic.plot_density import plot_density
from simulations.stochastic.set_target import set_target
from simulations.stochastic.set_init import set_default_init, set_noised_init

torch.set_default_dtype(torch.float64)

def main(target_name, proposal, divergence, alpha,
         num_samples, num_steps, lr, plot_folder, seed, device,
         scale_learnable, rand_init):
    set_seed(seed)

    if divergence in ["alpha", "alpha_reinforce"]:
        div_name = f"{divergence}_{alpha}"
    else:
        div_name = divergence

    folder_name = os.path.join(plot_folder,
                               f"{target_name}",
                               proposal,
                               div_name,
                               f"scale_learnable_{scale_learnable}",
                               f"run_{seed}_S_{num_samples}_E_{num_steps}_lr_{lr}")
    os.makedirs(folder_name, exist_ok=True)

    p = set_target(target_name, device)

    D = p.D # record dimension
    print("Target:", target_name, "D", D, "Num steps:", num_steps, "Num samples:", num_samples,
          "Div:", div_name)


    print("Target values:",
          "mean:", f"{p.mean}", "\n",
          "corr.:", f"{p.corr}", "\n",
          "cov.:", f"{p.cov}", "\n",
          )

    # plot target
    if D == 2:
        norm = plot_density(p, lower=-20, upper=20,
                 plt_title=os.path.join(folder_name, f"target.pdf"), steps=200)

    if rand_init == 'f':
        print("Using fixed init")
        q = set_default_init(target_name=target_name, proposal_name=proposal, D=D,
                         scale_learnable=scale_learnable, device=device, div=divergence, alpha=alpha)

    elif rand_init == 't':
        print("Using noised init")
        q = set_noised_init(target_name=target_name, proposal_name=proposal, D=D,
                             scale_learnable=scale_learnable, device=device, div=divergence, alpha=alpha)
    else:
        print("rand_init misspecified")

    print(f"Initial params:"
          f"\n mu={q.mean}, "
          f"\n corr={q.corr}",
          f"\n cov={q.cov}",
          )

    if scale_learnable == "t":
        optimizer = optim.Adam([q.mean, q.scale], lr=lr)
    else:
        optimizer = optim.Adam([q.mean], lr=lr)

    divs = []
    mean_errors = []
    corr_errors = []
    best_div = float("inf")
    for step in range(num_steps):
        optimizer.zero_grad()
        if divergence == "alpha":
            loss = alpha_div_reparam(p=p, q=q, S=num_samples, alpha=alpha)
        elif divergence == "alpha_reinforce":
            loss = alpha_div_reinforce(p=p, q=q, S=num_samples, alpha=alpha)
        elif divergence == "fkl":
            loss = fkl(p=p, q=q, S=num_samples)
        elif divergence == "rkl":
            loss = rkl_reparam(p=p, q=q, S=num_samples)
        elif divergence == "rkl_reinforce":
            loss = rkl_reinforce(p=p, q=q, S=num_samples)
        else:
            raise ValueError("Unknown div.")

        if torch.isnan(loss):
            print("Loss nan!")

        if torch.isinf(loss):
            print("Loss inf!")

        # get some stats
        with torch.no_grad():
            # compute errors
            mean_errors.append(delta_mean(p, q).detach().cpu())
            corr_errors.append(delta_corr(p, q).detach().cpu())

            # in case of reinforce append the ""actual loss'
            if divergence == "alpha_reinforce":
                loss_val = alpha_div_value(p=p, q=q, S=num_samples, alpha=alpha) # increased for MOU
            elif divergence == "rkl_reinforce":
                loss_val = rkl_value(p=p, q=q, S=num_samples)
            else:
                loss_val = loss

            if torch.isnan(loss_val):
                print("Loss val nan!")

            if torch.isinf(loss_val):
                print("Loss val inf!")

            # append div
            divs.append(loss_val.detach().cpu())

            if not (torch.isnan(loss_val) or torch.isinf(loss_val)):

                if loss_val.detach().cpu() < best_div and (loss_val.detach().cpu()) >= 0: # not instable
                    # if we have the best div save params
                    params_dict = {"proposal": proposal,
                               "D": p.D,
                               "mean_q": q.mean.detach().cpu().clone(),
                               "L": q.L.detach().cpu().clone(),
                                "scale_q": q.scale.detach().cpu().clone(),
                               "cov_q": q.cov.detach().cpu().clone(),
                               "corr_q": q.corr.detach().cpu().clone(),
                               "mean_p": p.mean.detach().cpu().clone(),
                               "corr_p": p.corr.detach().cpu().clone(),
                               "cov_p": p.cov.detach().cpu().clone(),
                               "target": target_name,
                               "scale_learnable": str(scale_learnable),
                               }
                    best_div = loss_val.detach().cpu()
                    print("Best model saved at step", step, "with loss", best_div)
            else:
                print("Skipping loss step due to inf/nan")

        # print stats every 1000 steps
        if step %  100 == 0:
            with torch.no_grad():
                print(f"Step {step}: Loss={loss.item():.3f}, Loss_val={loss_val.item():.3f}, "
                      f"\n mean_error={delta_mean(p, q).detach().cpu()}",
                      f"\n corr_error={delta_corr(p, q).detach().cpu()}",
                      "\n"
                      )
        if step % 1000 == 0:
            if D == 2:
                plot_density(q, lower=-20, upper=20, reference_mean=p.mean, norm=norm,
                             plt_title=os.path.join(folder_name, f"q_{step}.pdf"), steps=200)

        # do the backward
        try:
            loss.backward()

            # check for nans
            has_nan = False
            for param in q.parameters():
                if param.requires_grad:
                    if torch.isnan(param.grad).any():
                        has_nan = True
                        break

            if has_nan:
                print("NaNs in grad, skipping update")
                optimizer.zero_grad()
            else:
                optimizer.step()
        except:
            # if rare case with NaN, just skip the update
            print("Skipping step due to NaN")
            pass


    print("\n", "Training done, best loss:", best_div)
    plt.plot(range(num_steps), divs)
    plt.savefig(os.path.join(folder_name, "div.png"), bbox_inches="tight")
    plt.close()

    plt.plot(range(num_steps), mean_errors, label=r"$\Delta\mu$", linewidth=5)
    plt.plot(range(num_steps), corr_errors, label=r"$\Delta\text{Corr}$", linewidth=5)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(folder_name, "error.png"), bbox_inches="tight")
    plt.close()

    print("Starting to construct params dict...")
    print("Saving params dict...")
    torch.save(params_dict, os.path.join(folder_name, "model_weights.pt"))

    ### last
    if D == 2:
        plot_density(q, lower=-20, upper=20, reference_mean=p.mean, norm=norm,
                     plt_title=os.path.join(folder_name, f"q_last.pdf"), steps=200)

    params_dict_last = {"proposal": proposal,
                   "D": p.D,
                   "mean_q": q.mean.detach().cpu(),
                   "L": q.L.detach().cpu(),
                   "scale_q": q.scale.detach().cpu(),
                   "cov_q": q.cov.detach().cpu(),
                   "corr_q": q.corr.detach().cpu(),
                   "mean_p": p.mean.detach().cpu(),
                   "corr_p": p.corr.detach().cpu(),
                   "cov_p": p.cov.detach().cpu(),
                   "target": target_name,
                   "scale_learnable": str(scale_learnable),
                   }

    print("Starting to construct params dict (last)...")
    print("Saving params dict (last)...")
    torch.save(params_dict_last, os.path.join(folder_name, "model_weights_last.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name", type=str, help="name of the target density")
    parser.add_argument("--proposal", type=str, help="name of the variational family")
    parser.add_argument("--divergence", type=str, help="name of the divergence")
    parser.add_argument("--alpha", type=float, help="value of alpha for the alpha-DIV")
    parser.add_argument("--num_samples", type=int, help="number of samples used to compute the update")
    parser.add_argument("--num_steps", type=int, help="number of training steps")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--plot_folder", type=str, help="folder to store training plots")
    parser.add_argument("--seed", type=int, help="seed for reproducibility")
    parser.add_argument("--device", type=str, help="device for computations")
    parser.add_argument("--scale_learnable", type=str,
                        help="whether scale is learnable ('t') or not ('f')",
                        options=['t', 'f'])
    parser.add_argument("--rand_init",
                        type=str,
                        help="whether to noise the initialization ('t') or not ('f')",
                        options=['t', 'f'])

    args = parser.parse_args()

    main(target_name=args.target_name,
         proposal=args.proposal,
         divergence=args.divergence,
         alpha=args.alpha,
         num_samples=args.num_samples,
         num_steps=args.num_steps,
         lr=args.lr,
         plot_folder=args.plot_folder,
         seed=args.seed,
         device=args.device,
         scale_learnable=args.scale_learnable,
         rand_init=args.rand_init,
         )