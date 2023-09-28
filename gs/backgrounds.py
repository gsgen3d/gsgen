import random
import torch
import torch.nn as nn
from einops import repeat

tcnn_capable = True
try:
    import tinycudann as tcnn
except ImportError:
    tcnn_capable = False


class Background(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.random_aug = self.cfg.random_aug
        self.random_aug_prob = self.cfg.random_aug_prob

    def get_bg(self, dirs):
        raise NotImplementedError

    def forward(self, dirs):
        if not self.random_aug or (not self.training):
            return self.get_bg(dirs)
        else:
            if random.random() < self.random_aug_prob:
                return self.get_bg(dirs)
            else:
                return repeat(
                    torch.rand(3).to(dirs),
                    "c -> h w c",
                    h=dirs.shape[0],
                    w=dirs.shape[1],
                )


class FixedBackground(Background):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = cfg.device
        self.bg_color = nn.Parameter(torch.tensor(cfg.color))
        self.random_aug = False  # disable random augmentation
        self.random_aug_prob = 0.0  # make sure it is disabled

    def get_bg(self, dirs):
        H, W = dirs.shape[:2]
        return repeat(self.bg_color, "c -> h w c", h=H, w=W)


class RandomBackground(Background):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = cfg.device
        self.range = cfg.get("range", [0.0, 1.0])

    def get_bg(self, dirs):
        if self.training:
            color = torch.rand(3)
        else:
            color = torch.zeros(3)
        bg = repeat(
            color.to(dirs) * (self.range[1] - self.range[0]) + self.range[0],
            "c -> h w c",
            h=dirs.shape[0],
            w=dirs.shape[1],
        )

        return bg


class ConstBackground(Background):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = cfg.device
        self.initial_color = cfg.get("initial_color", [0.5, 0.5, 0.5])
        self.bg_color = nn.Parameter(self.initial_color)

    def get_bg(self, dirs):
        H, W = dirs.shape[:2]
        return repeat(self.bg_color, "c -> h w c", h=H, w=W)


class MLPBackground(Background):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.device = cfg.device
        if tcnn_capable:
            encoding_config = {"otype": "SphericalHarmonics", "degree": 3}
            network_config = {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 2,
            }
            self.mlp = tcnn.NetworkWithInputEncoding(
                3,
                3,
                encoding_config=encoding_config,
                network_config=network_config,
            )
        else:
            # add torch implementation of SH encoding and MLP
            raise ImportError("tinycudann not installed")

    def get_bg(self, dirs):
        return torch.nan_to_num(
            torch.sigmoid(self.mlp(dirs.reshape(-1, 3)).reshape(dirs.shape))
        )
