# Code for "Even more Guarantees for Variational Inference in the Presence of Symmetries''

This repository can be used to reproduce the simulations and figures in
***Even more Guarantees for Variational Inference in the Presence of Symmetries***,
presented at NeurIPS 2026.

### Repository structure
The repository has 2 main folders **figures** which reproduces core figures of the paper and **simulations**, which reproduces the experiments in section 4 of the paper.
The latter is further divided into a **symbolic** folder containing the experiments carried out by symbolic integration as described in section 4.1 and
**stochastic** which contains the experiments based on stochastic optimization presented in section 4.2.

### Running experiments
Please install the dependencies found in **requirements.txt**. 
To run an VI experiment, you can create a config file specifying the desired parameters and then use the script
simulations/stochastic/launch_from_config.py to execute it. We provide configs with the settings used in our experiments.
For instance, to reproduce our results on 2D targets, you can run the command below.
```
python -m simulations.stochastic.launch_from_config --config_path=simulations/stochastic/configs/2D_targets.yaml --device="cpu"
```
We also provide the model checkpoints in the folder **checkpoints**, which can be evaluated using the script **eval_table.py**.

To recreate the experiments with symbolic integration in Figure 4, you can run the scripts in the folder *simulations/symbolic*.
For instance, the command below runs the examples given for Theorem 1 and stores the resulting plots in the folder *figures/mean_recovery*
```
python -m simulations.symbolic.theorem_1
```

