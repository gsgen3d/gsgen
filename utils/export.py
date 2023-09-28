import numpy as np
import torch
from gs.gaussian_splatting import GaussianSplattingRenderer
from utils.ops import K_nearest_neighbors


def get_densify_val_grid(
    renderer: GaussianSplattingRenderer,
    L: float = -1.0,
    reso: int = 256,
    K: int = 3,
):
    if L < 0.0:
        L = renderer.mean.abs().max().item() * 1.1
    x = torch.linspace(-L, L, reso)
    y = torch.linspace(-L, L, reso)
    z = torch.linspace(-L, L, reso)

    x, y, z = torch.meshgrid(x, y, z)
    grid = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)

    mean = renderer.mean

    _, nn_idx = K_nearest_neighbors(grid, mean, K=K)

    densify_val_grid = torch.zeros_like(grid[..., :1])
