import os
import torch
import pandas as pd
from simulations.stochastic.vi_families import FCGProposal, FCTProposal, FCLProposal
from simulations.stochastic.targets import GMMTarget2D, GMMTarget4D, GMMTarget8D, GMMTarget16D, MOU, \
    Ellipse2D, Ellipse4D, Ellipse8D, Ellipse16D, AGMMTarget2D
from simulations.stochastic.metrics import delta_mean, delta_corr
from simulations.stochastic.plot_density import plot_density
import numpy as np
import argparse

def is_guaranteed(target, proposal, proposal_name, div, alpha=None):
    # returns whether we have a guarantee for the certain scenario
    mean_recov = False
    corr_recov = False
    theorems_mean = ""
    theorems_corr = ""

    ### mean recovery
    if target.is_even is True:
        if div == "fkl":
            if proposal.is_log_concave_strict:
                mean_recov = True
                theorems_mean += r"\ref{theorem:theorem_fkl_1}"
            elif target.support_cond and proposal.is_log_concave and proposal.is_rad_incr_strict:
                mean_recov = True
                theorems_mean += r"\ref{theorem:theorem_fkl_2}"
            else:
                print("No mean recovery guarantee found.")

        elif div in ["alpha", "alpha_reinforce"]:
            if alpha >= 2.0 and proposal_name != "l":
                mean_recov = True
                theorems_mean += r"\ref{theorem:theorem_alpha_1}"
                if target.support_cond:
                    theorems_mean += r",~\ref{theorem:theorem_alpha_2}"

    ### corr recovery
    if target.is_elliptic:
        assert target.is_even
        if div == "fkl":
            if mean_recov and proposal.is_log_concave:
                corr_recov = True
                theorems_corr += r"\ref{theorem:theorem_fkl_corr}"
        elif div in ["alpha", "alpha_reinforce"]:
            if mean_recov and proposal.is_log_concave and alpha > 1.0 and proposal_name != "l":
                corr_recov = True
                theorems_corr += r"\ref{theorem:theorem_alpha_corr}"

    theorems_mean = f"~({theorems_mean})"
    theorems_corr = f"~({theorems_corr})"
    return mean_recov, corr_recov, theorems_mean, theorems_corr


def is_guaranteed_rkl(target, proposal):
    # returns whether we have a guarantee when using the RKL
    mean_recov = False
    corr_recov = False
    if target.is_log_concave and target.is_even:
        mean_recov = True

    if target.is_elliptic and mean_recov:
        assert target.is_even
        corr_recov = True
    return mean_recov, corr_recov


def format_result(error, err_str, div, guarantee, vi_family, theorems=None):

    if guarantee:
        if div != "rkl" and error < 10 ** (-2):
            error_str = fr"$\textcolor{{cbg}}{{\boldsymbol{{{err_str}}}}}$"
            if vi_family == "g" and div == "fkl":
                error_str = fr"$\textcolor{{cbg}}{{\boldsymbol{{{err_str}}}}}^{{\boldsymbol{{*}}}}$"
            if theorems is not None:
                error_str += f" {theorems}"
        elif div != "rkl" and error >= 10 ** (-2):
            error_str = fr"${{\boldsymbol{{{err_str}}}}}$"
    else:
        if error < 10 ** (-2):
            error_str = fr"$\textcolor{{cbo}}{{\boldsymbol{{{err_str}}}}}$"
            if vi_family == "g" and div == "fkl":
                error_str = fr"$\textcolor{{cbg}}{{\boldsymbol{{{err_str}}}}}^{{\boldsymbol{{*}}}}$"
        else:
            error_str = fr"$\textcolor{{cbr}}{{\boldsymbol{{{err_str}}}}}$"
    return error_str

seeds = [42, 84, 126, 168, 210]

configs = {
    "mou2D":{
            "S": 10**5,
            "E": 10**4,
            "divs": ["rkl", "fkl", "alpha_reinforce"],
            "df_t": 5,
            "lr": 0.001,
            },

         "gmm2D":{
            "S": 10**5,
            "E": 10**4,
            "divs": ["rkl", "fkl", "alpha"],
             "df_t": 5,
            "lr": 0.001,
            },

        "ellipse2D":{
            "S": 10**5,
            "E": 10**4,
            "divs": ["rkl_reinforce", "fkl", "alpha"],
            "df_t": 5,
            "lr": 0.001,
            },

        "agmm2D":{
            "S": 10**5,
            "E": 10**4,
            "divs": ["rkl", "fkl", "alpha"],
            "df_t": 5,
            "lr": 0.001,
            },

        "gmm4D":{
            "S": 2*10**4,
            "E": 5*10**4,
            "divs": ["rkl", "fkl", "alpha"],
            "df_t": 5,
            "lr": 0.001,
            },

        "gmm8D":{
             "S": 2*10**4,
             "E": 5*10**4,
             "divs": ["rkl", "fkl", "alpha"],
             "df_t": 5,
            "lr": 0.001,
             },

        "gmm16D":{
            "S": 2*10**4,
            "E": 5*10**4,
            "divs": ["rkl", "fkl", "alpha"],
            "df_t": 5,
            "lr": 0.001,
            },

        "ellipse4D":{
            "S": 2*10**4,
            "E": 5*10**4,
            "divs": ["rkl_reinforce", "fkl", "alpha"],
            "df_t": 8,
            "lr": 0.01,
            },

        "ellipse8D":{
             "S": 2*10**4,
             "E": 5*10**4,
             "divs": ["rkl_reinforce", "fkl", "alpha"],
             "df_t": 8,
            "lr": 0.001,
             },

        "ellipse16D":{
            "S": 2*10**4,
            "E": 5*10**4,
            "divs": ["rkl_reinforce", "fkl", "alpha"],
            "df_t": 8,
            "lr": 0.001,
            },
}


proposals = ["g", "l", "t"]

proposal_names = {"g": "Gaussian",
                  "l": "Laplace",
                  "t": "Student-t"}

target_names_dict = {"mou2D": "MoU",
                     "ellipse2D": "Ellipse",
                     "gmm2D": "GMM",
                     "agmm2D": "A-GMM",
                     "gmm4D": "GMM (4D)",
                     "gmm8D": "GMM (8D)",
                     "gmm16D": "GMM (16D)",
                     "ellipse4D": "Ellipse (4D)",
                     "ellipse8D": "Ellipse (8D)",
                     "ellipse16D": "Ellipse (16D)",
                     }

def format_sci_latex(x, precision=1):
    s = f"{x:.{precision}e}"
    coeff, exp = s.split("e")
    exp = int(exp)  # remove leading zeros
    return rf"{coeff} \cdot 10^{{{exp}}}"


def eval_table(storing_loc_models, storing_loc_results):

    os.makedirs(storing_loc_results, exist_ok=True)

    all_rows = []
    for target_name in configs.keys():
        print(f"Starting {target_name} ...")
        if target_name == "gmm2D":
            p = GMMTarget2D("cpu")
            lower = -8
            upper = 12
        elif target_name == "ellipse2D":
            p = Ellipse2D(device="cpu")
            lower = -4
            upper = 6
        elif target_name == "mou2D":
            p = MOU(device="cpu")
            lower = -12
            upper = 15
        elif target_name == "agmm2D":
            p = AGMMTarget2D(device="cpu")
            lower = -10
            upper = 10
        elif target_name == "gmm4D":
            p = GMMTarget4D(device="cpu")
        elif target_name == "gmm8D":
            p = GMMTarget8D(device="cpu")
        elif target_name == "gmm16D":
            p = GMMTarget16D(device="cpu")
        elif target_name == "ellipse4D":
            p =  Ellipse4D(device="cpu")
        elif target_name == "ellipse8D":
            p =  Ellipse8D(device="cpu")
        elif target_name == "ellipse16D":
            p =  Ellipse16D(device="cpu")
        else:
            raise ValueError("Target misspecified.")

        # plot the target
        if p.D == 2:
            norm = plot_density(density=p, lower=lower, upper=upper, plt_title=os.path.join(storing_loc_results, f"{target_name}_target.pdf"))

        target_rows = []
        for i, vi_family in enumerate(proposals):
            if vi_family == 'l' and p.D > 2:
                print("Skipping Laplace.")
                continue
            prop_row = []
            if i == 0:
                if p.D ==2:
                    prop_row.extend([fr"\multirow{{3}}{{*}}{{{target_names_dict[target_name]}}}", proposal_names[vi_family]])
                else:
                    prop_row.extend([fr"\multirow{{2}}{{*}}{{{target_names_dict[target_name]}}}", proposal_names[vi_family]])

            else:
                prop_row.extend([" ", proposal_names[vi_family]])
            print("\n")
            for div in configs[target_name]['divs']:
                if div in ["alpha", "alpha_reinforce"]:
                    alphas = [0.5, 2.0]
                else:
                    alphas = [0.0]

                if div == "rkl" and target_name == "mou2D":
                    prop_row.extend(["/", "/"])
                    continue

                for alpha in alphas:
                    all_errors_mean = []
                    all_errors_corr = []
                    for s in seeds:
                        if div in ["alpha", "alpha_reinforce"]:
                            model_path = os.path.join(storing_loc_models,
                                                  target_name,
                                                  vi_family,
                                                  f"{div}_{alpha}",
                                                  "scale_learnable_t",
                                                  f"run_{s}_S_{configs[target_name]['S']}_E_{configs[target_name]['E']}_lr_{configs[target_name]['lr']}")
                        else:
                            model_path = os.path.join(storing_loc_models,
                                                      target_name,
                                                      vi_family,
                                                      div,
                                                      "scale_learnable_t",
                                                      f"run_{s}_S_{configs[target_name]['S']}_E_{configs[target_name]['E']}_lr_{configs[target_name]['lr']}")

                        if target_name == "mou2D" and div == "alpha_reinforce":
                            weights = torch.load(os.path.join(model_path, "model_weights_last.pt"))
                        else:
                            weights = torch.load(os.path.join(model_path, "model_weights.pt"))
                        expected_mean = weights['mean_q']
                        expected_cov = weights['cov_q']
                        expected_L = weights['L']
                        scale = weights['scale_q']


                        with torch.no_grad():
                            if vi_family == "g":
                                q = FCGProposal(mean=weights['mean_q'], scale=scale, device="cpu", scale_learnable=True)
                                error_mean = delta_mean(p=p, q=q)
                                error_corr = delta_corr(p=p, q=q)

                            elif vi_family == "l":
                                q = FCLProposal(mean=weights['mean_q'], scale=scale, device="cpu", scale_learnable=True)
                                error_mean = delta_mean(p=p, q=q)
                                error_corr = delta_corr(p=p, q=q)

                            elif vi_family == "t":
                                q = FCTProposal(mean=weights['mean_q'], scale=scale, device="cpu", df=configs[target_name]['df_t'], scale_learnable=True)
                                error_mean = delta_mean(p=p, q=q)
                                error_corr = delta_corr(p=p, q=q)
                            else:
                                raise ValueError("vi family misspecified")

                            # ensure model is loaded correctly
                            assert torch.allclose(expected_mean, q.mean)
                            assert torch.allclose(expected_cov, q.cov)
                            assert torch.allclose(expected_L, q.L)

                        all_errors_mean.append(error_mean)
                        print(vi_family, div, alpha, error_mean)
                        all_errors_corr.append(error_corr)

                    if div != "rkl":
                        mean_guarantee, corr_guarantee, theorems_mean, theorems_corr = is_guaranteed(target=p,
                                                                                                     proposal=q,
                                                                                                     proposal_name=vi_family,
                                                                                                     div=div,
                                                                                                     alpha=alpha)
                    else:
                        mean_guarantee, corr_guarantee = is_guaranteed_rkl(p, q)
                        print(mean_guarantee, corr_guarantee)

                    error_mean = sum(all_errors_mean)/len(all_errors_mean)
                    error_corr = sum(all_errors_corr) / len(all_errors_corr)
                    stdv_error_mean = np.std(all_errors_mean, ddof=1)
                    stdv_error_corr = np.std(all_errors_corr, ddof=1)

                    mean_err_str = format_sci_latex(error_mean)
                    corr_err_str = format_sci_latex(error_corr)
                    stdv_mean_err_str = f"{{\\scriptsize \pm {format_sci_latex(stdv_error_mean)}}}"
                    stdv_corr_err_str = f"{{\\scriptsize \pm {format_sci_latex(stdv_error_corr)}}}"

                    mean_err_str = f"{mean_err_str} {stdv_mean_err_str}"
                    corr_err_str = f"{corr_err_str} {stdv_corr_err_str}"

                    mean_str = format_result(error=error_mean, err_str=mean_err_str, div=div,
                                             guarantee=mean_guarantee, vi_family=vi_family)

                    corr_str = format_result(error=error_corr, err_str=corr_err_str, div=div,
                                             guarantee=corr_guarantee, vi_family=vi_family)

                    prop_row.extend([mean_str, corr_str])
            all_rows.append(prop_row)
        all_rows.extend(target_rows)

    print("Full table:",)
    df = pd.DataFrame(all_rows)
    df_latex = df.to_latex(escape=False, index=False)

    lines = df_latex.splitlines()

    insert_after = [4, 7, 10, 13, 15, 17, 19, 21, 23]

    new_lines = []
    row_counter = 0

    for line in lines:
        new_lines.append(line)
        if line.strip().endswith(r'\\'):  # detects a data row
            row_counter += 1
            if row_counter in insert_after:
                new_lines.append(r'\midrule')

    latex_with_midrules = "\n".join(new_lines)

    print(latex_with_midrules)

    # small table with a single run
    s = seeds[0]
    target_names = ["mou2D", "gmm2D", "ellipse2D", "agmm2D"]

    combs_per_target = {
        "mou2D": [("fkl", 0.0, "g"), ("fkl", 0.0, "t"), ("alpha", 2.0, "t"), ("alpha", 0.5, "t")],
        "gmm2D": [("fkl", 0.0, "g"), ("fkl", 0.0, "t"), ("alpha", 2.0, "t"), ("alpha", 0.5, "t")],
        "ellipse2D": [("fkl", 0.0, "g"), ("fkl", 0.0, "t"), ("alpha", 2.0, "g"), ("alpha", 0.5, "t")],
        "agmm2D": [("fkl", 0.0, "g"), ("fkl", 0.0, "t"), ("alpha", 2.0, "g"), ("alpha", 0.5, "t")],
    }

    all_rows = []
    for target_name in target_names:
        if target_name == "gmm2D":
            p = GMMTarget2D("cpu")
            lower = -8
            upper = 12
        elif target_name == "ellipse2D":
            p = Ellipse2D(device="cpu")
            lower = -4
            upper = 6
        elif target_name == "mou2D":
            p = MOU(device="cpu")
            lower = -12
            upper = 15
        elif target_name == "agmm2D":
            p = AGMMTarget2D(device="cpu")
            lower = -10
            upper = 10
        else:
            raise ValueError("Target misspecified.")
        target_row_m = []
        target_row_c = []
        for div, alpha, vi_family in combs_per_target[target_name]:
            if target_name == "mou2D" and div == "alpha":
                div = "alpha_reinforce"
            if div in ["alpha", "alpha_reinforce"]:
                model_path = os.path.join(storing_loc_models,
                                          target_name,
                                          vi_family,
                                          f"{div}_{alpha}",
                                          "scale_learnable_t",
                                          f"run_{s}_S_{configs[target_name]['S']}_E_{configs[target_name]['E']}_lr_{configs[target_name]['lr']}")
            else:
                model_path = os.path.join(storing_loc_models,
                                          target_name,
                                          vi_family,
                                          div,
                                          "scale_learnable_t",
                                          f"run_{s}_S_{configs[target_name]['S']}_E_{configs[target_name]['E']}_lr_{configs[target_name]['lr']}")

            if target_name == "mou2D" and div == "alpha_reinforce":
                # for stability
                print("Loading last checkpoint for MOU with alpha-div!")
                weights = torch.load(os.path.join(model_path, "model_weights_last.pt"))
            else:
                weights = torch.load(os.path.join(model_path, "model_weights.pt"))
            expected_mean = weights['mean_q']
            expected_cov = weights['cov_q']
            expected_L = weights['L']
            scale = weights['scale_q']

            with torch.no_grad():
                if vi_family == "g":
                    q = FCGProposal(mean=weights['mean_q'], scale=scale, device="cpu", scale_learnable=True)
                    error_mean = delta_mean(p=p, q=q)
                    error_corr = delta_corr(p=p, q=q)

                elif vi_family == "l":
                    q = FCLProposal(mean=weights['mean_q'], scale=scale, device="cpu", scale_learnable=True)
                    error_mean = delta_mean(p=p, q=q)
                    error_corr = delta_corr(p=p, q=q)

                elif vi_family == "t":
                    q = FCTProposal(mean=weights['mean_q'], scale=scale, device="cpu", df=configs[target_name]['df_t'], scale_learnable=True)
                    error_mean = delta_mean(p=p, q=q)
                    error_corr = delta_corr(p=p, q=q)

                else:
                    raise ValueError("vi family misspecified")

                assert torch.allclose(expected_mean, q.mean)
                assert torch.allclose(expected_cov, q.cov)
                assert torch.allclose(expected_L, q.L)
                print(target_name, vi_family, error_mean)

                if div != "rkl":
                    mean_guarantee, corr_guarantee, theorems_mean, theorems_corr = is_guaranteed(target=p,
                                                                                                 proposal=q,
                                                                                                 proposal_name=vi_family,
                                                                                                 div=div,
                                                                                                 alpha=alpha)
                else:
                    mean_guarantee, corr_guarantee = is_guaranteed_rkl(p, q)
                    print(target_name, mean_guarantee, corr_guarantee)

                mean_err_str = format_sci_latex(error_mean)
                corr_err_str = format_sci_latex(error_corr)

                mean_err_str = f"{mean_err_str}"
                corr_err_str = f"{corr_err_str}"

                mean_str = format_result(error=error_mean, err_str=mean_err_str, div=div,
                                         guarantee=mean_guarantee, theorems=theorems_mean, vi_family=vi_family)
                corr_str = format_result(error=error_corr, err_str=corr_err_str, div=div,
                                         guarantee=corr_guarantee, theorems=theorems_corr, vi_family=vi_family)

                target_row_m.extend([mean_str])
                target_row_c.extend([corr_str])

                ### plot the density
                if p.D == 2:
                    if vi_family == "l" and target_name == "mou2D" and alpha == 0.5:
                        # printed without norm
                        print("Plotted without norm!")
                        plot_density(density=q, lower=lower, upper=upper,
                                     plt_title=os.path.join(storing_loc_results,
                                                            f"{target_name}_{vi_family}_{div}_{alpha}.pdf"),
                                     norm=None,
                                     reference_mean=p.mean)
                    else:
                        plot_density(density=q, lower=lower, upper=upper,
                                     plt_title=os.path.join(storing_loc_results,
                                                            f"{target_name}_{vi_family}_{div}_{alpha}.pdf"),
                                     norm=norm,
                                     reference_mean=p.mean)

        all_rows.append(target_row_m)
        all_rows.append(target_row_c)


    print("Small table:",)
    df_small = pd.DataFrame(all_rows)
    print(df_small.to_latex(escape=False, index=False))
    print("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--storing_loc_models", type=str, help="where the models are stored")
    parser.add_argument("--storing_loc_results", type=str, help="where any resulting plots are stored")
    args = parser.parse_args()

    eval_table(
        storing_loc_models=args.storing_loc_models,
        storing_loc_results=args.storing_loc_results
    )

