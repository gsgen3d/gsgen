import torch
import numpy as np
from scipy.special import logit, expit


def wrapper(number_fn, tensor_fn):
    def wrapped(x):
        if isinstance(x, torch.Tensor):
            return tensor_fn(x)
        else:
            return number_fn(x)

    return wrapped


min_scale = 1e-3


def biased_relu(x):
    return torch.relu(x) + min_scale


def biased_abs(x):
    return torch.abs(x) + min_scale


def biased_abs_inv(x):
    return torch.abs(x - min_scale)


softplus_inv_numeric = lambda x: np.log(np.expm1(x))


def softplus_inv(x):
    return x + torch.log(-torch.expm1(-x))


activations = dict(
    abs=torch.abs,
    relu=torch.nn.functional.relu,
    sigmoid=torch.sigmoid,
    nothing=lambda x: x,
    exp=torch.exp,
    biased_relu=biased_relu,
    biased_abs=biased_abs,
    softplus=torch.nn.functional.softplus,
)

inv_activations = dict(
    abs=wrapper(np.abs, torch.abs),
    nothing=lambda x: x,
    sigmoid=wrapper(logit, torch.logit),
    relu=lambda x: x,
    exp=wrapper(np.log, torch.log),
    biased_relu=lambda x: x - min_scale,
    biased_abs=lambda x: x - min_scale,
    softplus_inv=wrapper(softplus_inv_numeric, softplus_inv),
)
