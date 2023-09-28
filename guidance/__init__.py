import torch


class BaseGuidance(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = self.cfg.device

    def forward(self, image):
        raise NotImplementedError("Guidance forward not implemented.")

    def update(self, step):
        raise NotImplementedError("Guidance update not implemented.")

    def text2img(self, text):
        pass

    def log(self, writer, step):
        pass


from .stable_diffusion import StableDiffusionGuidance
from .stable_diffusion_vsd import StableDiffusionVSDGuidance
from .deep_floyd import DeepFloydGuidance
from .point_e import PointEGuidance
from .make_it_3d import MakeIt3DGuidance

guidances = dict(
    stable_diffusion=StableDiffusionGuidance,
    deep_floyd=DeepFloydGuidance,
    point_e=PointEGuidance,
    stable_diffusion_vsd=StableDiffusionVSDGuidance,
    make_it_3d=MakeIt3DGuidance,
)


def get_guidance(cfg):
    try:
        return guidances[cfg.type](cfg)
    except KeyError:
        raise NotImplementedError(f"Guidance {cfg.type} not implemented.")
