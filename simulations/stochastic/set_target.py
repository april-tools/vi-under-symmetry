from simulations.stochastic.targets import *

def set_target(target_name, device):
    if target_name == "gmm2D":
        p = GMMTarget2D(device=device)
    elif target_name == "gmm4D":
        p = GMMTarget4D(device=device)
    elif target_name == "gmm8D":
        p = GMMTarget8D(device=device)
    elif target_name == "gmm16D":
        p = GMMTarget16D(device=device)
    elif target_name == "agmm2D":
        p = AGMMTarget2D(device=device)
    elif target_name == "mou2D":
        p = MOU(device=device)
    elif target_name == "ellipse2D":
        p = Ellipse2D(device=device)
    elif target_name == "ellipse4D":
        p = Ellipse4D(device=device)
    elif target_name == "ellipse8D":
        p = Ellipse8D(device=device)
    elif target_name == "ellipse16D":
        p = Ellipse16D(device=device)
    else:
        raise ValueError("Target missepecified.")
    return p