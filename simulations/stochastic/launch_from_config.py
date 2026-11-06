from simulations.stochastic.single_run import main
import argparse
import yaml

def run_exp(seed, target_name, div_names,  device, plot_folder, num_samples, num_steps, lr, proposal, scale_learnable, rand_init, n_reps):
    for target_n in target_name:
        for p in proposal:
            for div in div_names:
                if div not in ["alpha", "alpha_reinforce"]:
                    alpha_l = [0.0]
                else:
                    alpha_l = [0.5, 2.0]
                for alpha in alpha_l:
                    for i, rep in enumerate(range(n_reps)):
                        seed_c = seed + i * 42
                        print("Seed", seed_c)
                        main(target_name=target_n,
                                proposal=p,
                                divergence=div,
                                alpha=alpha,
                                num_samples=num_samples,
                                num_steps=num_steps,
                                lr=lr,
                                plot_folder=plot_folder,
                                seed=seed_c,
                                device=device,
                                scale_learnable=scale_learnable,
                                rand_init=rand_init,
                            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        help="path to config file that should be used")
    parser.add_argument("--storing_loc", type=str,
                        help="path to storing location for models")
    parser.add_argument("--device", type=str,
                        help="device for experiments")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    run_exp(
        target_name=cfg["target_name"],
        div_names=cfg["div_names"],
        device=args.device,
        plot_folder=args.storing_loc,
        seed=cfg["seed"],
        num_samples=cfg["num_samples"],
        num_steps=cfg["num_steps"],
        lr=cfg["lr"],
        proposal=cfg["proposal"],
        scale_learnable=cfg["scale_learnable"],
        rand_init=cfg["rand_init"],
        n_reps=cfg["n_reps"]
    )