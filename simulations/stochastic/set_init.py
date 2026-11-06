"""
Initialize the variational family.
"""
import torch
from simulations.stochastic.vi_families import FCGProposal, FCTProposal, FCLProposal

def scale_from_cov(cov):
    # inverts the scaling done to ensure pos def preprocess
    L_target = torch.linalg.cholesky(cov)

    scale = L_target.clone()

    diag = torch.diagonal(L_target)
    scale = scale - torch.diag(diag) + torch.diag(torch.log(diag))

    return scale

def student_t_scale_from_cov(cov, df):
    cov = cov * torch.tensor((df - 2) / df)
    L_target = torch.linalg.cholesky(cov)

    scale = L_target.clone()

    diag = torch.diagonal(L_target)
    scale = scale - torch.diag(diag) + torch.diag(torch.log(diag))
    return scale

def get_cov_init(D, target_name, div, alpha, proposal):
    # construct the initial covariance
    cov_init = torch.zeros(D, D)  # initialize the scale matrix

    if target_name in ["gmm2D", "studentMM", "mou2D", "agmm2D", "gaussian2D"]:
        init_var = torch.tensor(25)
        cov_init.fill_diagonal_(init_var)

    elif target_name in ["ellipse2D", "ellipse4D", "ellipse8D"]:
        init_var = torch.tensor(25)
        cov_init.fill_(init_var)
        cov_init.fill_diagonal_(30)

    elif target_name == "ellipse16D":
        if proposal == "t":
            init_var = torch.tensor(1)
            cov_init.fill_(init_var)
            cov_init.fill_diagonal_(3)
        elif proposal == "g":
            init_var = torch.tensor(1.8)
            cov_init.fill_(init_var)
            cov_init.fill_diagonal_(3) # works but very close
        else:
            print("proposal misspecified")

    elif target_name in ["gmm4D", "gmm8D", "gmm16D"]:
        init_var = torch.tensor(3)
        cov_init.fill_(init_var)
        cov_init.fill_diagonal_(4)

    else:
        raise ValueError("Target not defined.")
    return cov_init

def set_default_init(target_name, proposal_name, D, scale_learnable, device, div, alpha):
    mean = torch.zeros(D)

    if scale_learnable == "t":
        scale_learnable = True
    elif scale_learnable == "f":
        scale_learnable = False
    else:
        raise ValueError("Scale learnable misspecified.")

    cov_init = get_cov_init(D, target_name, div, alpha, proposal_name)

    if proposal_name == "g":
        sqrt_S = scale_from_cov(cov_init)
        q = FCGProposal(mean=mean, scale=sqrt_S, device=device, scale_learnable=scale_learnable)
    elif proposal_name == "t":
        if target_name in ["ellipse4D", "ellipse8D", "ellipse16D"]:
            df = 8
        else:
            df = 5
        print("Student-t used with", df, "degrees od freedom")
        sqrt_S = student_t_scale_from_cov(cov_init, df)
        q = FCTProposal(mean=mean, scale=sqrt_S, device=device,
                        df=df, scale_learnable=scale_learnable)
    elif proposal_name == "l":
        sqrt_S = scale_from_cov(cov_init)
        q = FCLProposal(mean=mean, scale=sqrt_S, device=device, scale_learnable=scale_learnable)
    else:
        raise ValueError(proposal_name, "Unknown proposal.")
    return q


def set_noised_init(target_name, proposal_name, D, scale_learnable, device, div, alpha):
    mean = torch.zeros(D)
    if D < 16:
        mean += torch.randn(mean.shape[0])
    else:
        mean += 0.1*torch.randn(mean.shape[0])

    if target_name in ["ellipse4D", "ellipse8D", "ellipse16D"]:
        df = 8
    else:
        df = 5

    if scale_learnable == "t":
        scale_learnable = True
    elif scale_learnable == "f":
        scale_learnable = False
    else:
        raise ValueError("Scale learnable misspecified.")

    cov_init = get_cov_init(D, target_name, div, alpha, proposal_name)
    if proposal_name != "t":
        L = student_t_scale_from_cov(cov_init, df=df)
    else:
        L = scale_from_cov(cov_init)
    L += torch.tril(0.2*torch.rand(D, D))
    diag_raw = torch.diagonal(L)
    diag_pos = torch.exp(diag_raw)  # exp the diag
    L = L - torch.diag(diag_raw) + torch.diag(diag_pos)
    cov_init = L@L.T

    if proposal_name == "g":
        sqrt_S = scale_from_cov(cov_init)
        q = FCGProposal(mean=mean, scale=sqrt_S, device=device, scale_learnable=scale_learnable)
    elif proposal_name == "t":
        print("Student-t used with", df, "degrees od freedom")
        sqrt_S = student_t_scale_from_cov(cov_init, df)
        q = FCTProposal(mean=mean, scale=sqrt_S, device=device,
                        df=df, scale_learnable=scale_learnable)
    elif proposal_name == "l":
        sqrt_S = scale_from_cov(cov_init)
        q = FCLProposal(mean=mean, scale=sqrt_S, device=device, scale_learnable=scale_learnable)
    else:
        raise ValueError(proposal_name, "Unknown proposal.")
    return q