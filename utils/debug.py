import torch
from omegaconf import OmegaConf


def debug_initialize(debug_flag, cfg=None):
    initial_values = {}
    if debug_flag == "one":
        # A big gaussian in the center
        initial_values["mean"] = torch.tensor([[0.0, 0.0, 0.0]])
        initial_values["svec"] = torch.tensor([[0.1, 0.1, 0.2]])
        initial_values["qvec"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        initial_values["color"] = torch.tensor([[0.01, 0.01, 0.99]])
        initial_values["alpha"] = torch.tensor([0.8])
    elif debug_flag == "two":
        # A big gaussian in the center
        initial_values["mean"] = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.4, 0.0]])
        initial_values["svec"] = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        initial_values["qvec"] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        )
        initial_values["color"] = torch.tensor([[0.01, 0.01, 0.99], [0.01, 0.01, 0.99]])
        initial_values["alpha"] = torch.tensor([0.8, 0.8])
    elif debug_flag == "pressure":
        n_points = cfg.n_points
        bounds = cfg.bounds
        initial_values["mean"] = torch.randn((n_points, 3)) * bounds
        initial_values["svec"] = torch.rand((n_points, 3)) * 0.05
        initial_values["qvec"] = torch.rand((n_points, 4))
        initial_values["color"] = torch.rand((n_points, 3))
        initial_values["alpha"] = torch.rand((n_points))
    elif debug_flag == "paper":
        initial_values["mean"] = torch.tensor([[0.0, -0.3, 0.2], [0.0, 0.3, -0.1]])
        initial_values["svec"] = torch.tensor([[0.1, 0.2, 0.1], [0.1, 0.1, 0.2]])
        initial_values["qvec"] = torch.FloatTensor([[1, 1, 0, 1], [1, 0, 1, 0]])
        initial_values["color"] = torch.tensor([[0.0, 0.0, 0.9], [0.0, 0.0, 0.9]])
        initial_values["alpha"] = torch.tensor([0.9, 0.9])
    else:
        raise NotImplementedError

    return initial_values


def debug_optimizer_cfg(override=None):
    cfg = {
        "type": "Adam",
        "opt_args": {
            "eps": 1e-15,
        },
    }
    if override is not None:
        cfg.update(override)

    return OmegaConf.create(cfg)


def debug_lr_cfg(override=None):
    cfg = {
        "mean": 3e-3,
        "svec": 3e-3,
        "qvec": 3e-3,
        "color": 3e-3,
        "alpha": 3e-3,
        "bg": 3e-3,
    }
    if override is not None:
        cfg.update(override)

    return OmegaConf.create(cfg)
