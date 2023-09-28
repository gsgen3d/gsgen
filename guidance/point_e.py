import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from point_e.models.configs import model_from_config, MODEL_CONFIGS
from point_e.diffusion.configs import diffusion_from_config, DIFFUSION_CONFIGS
from point_e.diffusion.k_diffusion import get_sigmas_karras
from point_e.models.download import load_checkpoint
from utils.misc import print_info
from utils.ops import farthest_point_sampling

POINT_E_NUM_POINTS = 1024


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


class PointEGuidance(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device = cfg.device
        self.device = cfg.device
        self.cfg = cfg
        self.base_name = cfg.base_name
        model_cfg = MODEL_CONFIGS[self.base_name]
        diffusion_cfg = DIFFUSION_CONFIGS[self.base_name]
        diffusion_cfg["channel_scales"] = [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
        diffusion_cfg["channel_biases"] = [
            0.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            -1.0,
        ]
        self.model = model_from_config(model_cfg, device)
        self.diffusion = diffusion_from_config(diffusion_cfg)
        self.timesteps = diffusion_cfg["timesteps"]
        self.weighting_strategy = cfg["weighting_strategy"]
        self.device = device
        self.model.load_state_dict(load_checkpoint(cfg.base_name, device))

        self.scheduler_type = cfg.scheduler_type
        if self.scheduler_type == "original":
            self.alphas = (
                torch.from_numpy(self.diffusion.alphas_cumprod)
                .to(device)
                .to(torch.float32)
            )
        elif self.scheduler_type == "karras":
            # TODO
            from point_e.diffusion.k_diffusion import (
                get_sigmas_karras,
                GaussianToKarrasDenoiser,
            )

            model = self.model
            del self.model
            self.model = GaussianToKarrasDenoiser(model, self.diffusion)
            self.sigmas = get_sigmas_karras(
                self.timesteps,
                cfg.sigma_min,
                cfg.sigma_max,
                cfg.rho,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        self.text = None

    def set_text(self, text: str):
        self.text = text

    def get_epsilon(self, x, t, noise, model_kwargs):
        x = self.diffusion.scale_channels(x)

        x_t = self.diffusion.q_sample(x, t, noise)
        t_ = torch.cat([t, t], dim=0)
        x_t = torch.cat([x_t, x_t], dim=0)
        if self.scheduler_type == "karras":
            sigma_ = self.sigmas[t_]
            t_ = torch.tensor(
                [self.model.sigma_to_t(sigma) for sigma in sigma_.cpu().numpy()],
                dtype=torch.long,
                device=self.sigmas.device,
            )
            c_in = append_dims(1.0 / (sigma_**2 + 1) ** 0.5, x_t.ndim)
            out = self.diffusion.p_mean_variance(
                self.model.model,
                x_t * c_in,
                t_,
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )["model_output"]
        elif self.scheduler_type == "original":
            out = self.model(x_t, t_, **model_kwargs)

        cond, uncond = out.chunk(2, dim=0)

        eps = uncond + self.cfg.guidance_scale * (cond - uncond)

        return eps

    def forward_image(self, x, images=None):
        # x: [L, C]
        bs = self.cfg.batch_size

        x = repeat(x, "L C -> B L C", B=bs)
        _, indices = farthest_point_sampling(x, POINT_E_NUM_POINTS, True)
        x = x[indices]
        x = x.moveaxis(1, -1)
        t = torch.randint(
            int(self.timesteps * self.cfg.min_step_percent),
            int(self.timesteps * self.cfg.max_step_percent),
            (bs,),
            device=x.device,
            dtype=torch.long,
        )

        # TODO: check here
        noise = torch.randn_like(x)
        noise_pred = self.get_epsilon(x, t, noise, {"images": images + [None] * bs})

        if self.weighting_strategy == "sds":
            # w(t), sigma_t^2
            if self.scheduler_type == "original":
                w = (1 - self.alphas[t]).view(-1, 1, 1)
            elif self.scheduler_type == "karras":
                w = (1 - self.sigmas[t]).view(-1, 1, 1)
        elif self.weighting_strategy == "uniform":
            w = 1
        elif self.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

        # print(noise.shape)
        # print(noise_pred.shape)
        # noise_pred contains variance
        grad = w * (noise_pred[:, :6] - noise)

        grad = torch.nan_to_num(grad)
        # # clip grad for stable training?
        # if self.grad_clip_val is not None:
        #     grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        target = (x - grad).detach()

        loss_sds = 0.5 * F.mse_loss(x, target, reduction="sum") / bs

        return loss_sds

    def forward_text(self, xyz, rgb, texts=None):
        # breakpoint()
        bs = self.cfg.batch_size

        x = torch.cat([xyz, rgb], dim=-1)
        x = repeat(x, "L C -> B L C", B=bs)
        _, indices = farthest_point_sampling(x[..., :3], POINT_E_NUM_POINTS, True)
        x_ = []
        for x_i, indices_i in zip(x, indices):
            x_.append(x_i[indices_i])
        x = torch.stack(x_, dim=0)
        # breakpoint()

        x = x.moveaxis(1, -1)
        t = torch.randint(
            int(self.timesteps * self.cfg.min_step_percent),
            int(self.timesteps * self.cfg.max_step_percent),
            (bs,),
            device=x.device,
            dtype=torch.long,
        )

        # TODO: check here
        noise = torch.randn_like(x)
        if texts is None:
            texts = [self.text] * bs
        noise_pred = self.get_epsilon(x, t, noise, {"texts": texts + [None] * bs})

        if self.weighting_strategy == "sds":
            # w(t), sigma_t^2
            if self.scheduler_type == "original":
                w = (1 - self.alphas[t]).view(-1, 1, 1)
            elif self.scheduler_type == "karras":
                w = (1 - self.sigmas[t]).view(-1, 1, 1)
        elif self.weighting_strategy == "uniform":
            w = 1
        elif self.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1)
        else:
            raise ValueError(f"Unknown weighting strategy: {self.weighting_strategy}")

        # print(noise.shape)
        # print(noise_pred.shape)
        # noise_pred contains variance
        grad = w * (noise_pred[:, :6] - noise)

        grad = torch.nan_to_num(grad)
        # # clip grad for stable training?
        # if self.grad_clip_val is not None:
        #     grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        target = (x - grad).detach()

        loss_sds = 0.5 * F.mse_loss(x, target, reduction="sum") / bs

        return loss_sds

    def forward(self, renderer):
        # True for point-e loss only apply on mean of gaussians
        assert self.text is not None, "Please set text prompt first"
        xyz = renderer.mean
        rgb = renderer.color
        if self.cfg.normalize:
            xyz = xyz / xyz.norm(dim=-1).max().detach() * 0.5
        if self.cfg.mean_only:
            rgb = rgb.detach()
        return self.forward_text(xyz, rgb)
