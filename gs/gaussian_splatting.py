import gc
from typing import Any
import cv2
import warnings
from pathlib import Path
from utils.camera import PerspectiveCameras
from utils.transforms import qsvec2rotmat_batched, qvec2rotmat_batched
from utils.misc import print_info, tic, toc, list_to_float
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from rich.console import Console
from utils.activations import activations, inv_activations
from gs.culling import tile_culling_aabb_count
from utils.misc import lineprofiler, to_primitive, stack_dicts, C, C_wrapped
from utils.ops import (
    estimate_normal,
    nearest_neighbor,
    distance_to_gaussian_surface,
    K_nearest_neighbors,
    compute_shaded_color,
)
from timeit import timeit
from time import time
from omegaconf import OmegaConf

# from .initialize import cov_init, alpha_center_anealing_init, alpha_trunc_init
from utils.schedulers import lr_schedulers
from utils.optimizers import optimizers
from .backgrounds import (
    RandomBackground,
    ConstBackground,
    MLPBackground,
    FixedBackground,
)

console = Console()

try:
    import _gs as _backend
except ImportError:
    console.print("Existing installation not fonud")
    from .backend import _backend

from gs.renderer import (
    render_sh,
    step_check,
    project_gaussians,
    render_start_end,
    render_sh_bg,
    render_scalar,
    render_with_T,
)

field2raw = dict(
    mean="mean",
    qvec="qvec",
    svec="svec_before_activation",
    color="color_before_activation",
    alpha="alpha_before_activation",
    normal="normal_before_activation",
)


class GaussianSplattingRenderer(torch.nn.Module):
    def __init__(self, cfg, initial_values=None):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device

        self.pbr = self.cfg.get("pbr", False)
        self.normal_type = self.cfg.get("normal_type", "estimated")

        self.svec_act = activations[cfg.svec_act]
        self.alpha_act = activations[cfg.alpha_act]
        self.color_act = activations[cfg.color_act]

        self.svec_inv_act = inv_activations[cfg.svec_act]
        self.alpha_inv_act = inv_activations[cfg.alpha_act]
        self.color_inv_act = inv_activations[cfg.color_act]

        self.step = 0
        self.N = -1
        if initial_values is not None:
            self.initialize(initial_values)

        self.N_changed = False

        if hasattr(self, "mean"):
            self.register_buffer("grad_mean2d", torch.zeros_like(self.mean[..., 0]))
            self.register_buffer(
                "cnt", torch.zeros_like(self.mean[..., 0], dtype=torch.int32)
            )
        self.setup(cfg)

        # for densify log
        self.masks = []
        self.mean_2ds = []

        self.mask_start_step = self.cfg.get("mask_start_step", -1)
        self.mask_end_step = self.cfg.get("mask_end_step", -1)
        self.mask_enabled = self.mask_start_step >= 0 and self.mask_end_step >= 0
        if self.mask_enabled:
            console.print(
                "[red]gradient mask enabled, which is an experimental feature for single image to 3d"
            )

        self.to(self.device)

    @property
    def svec(self):
        return self.svec_act(self.svec_before_activation)

    @property
    def alpha(self):
        return self.alpha_act(self.alpha_before_activation)

    @property
    def color(self):
        return self.color_act(self.color_before_activation)

    @svec.setter
    def svec(self, value):
        if value.shape == self.svec_before_activation.shape:
            self.svec_before_activation.data = self.svec_inv_act(value)
        else:
            self.svec_before_activation = nn.Parameter(self.svec_inv_act(value.data))

    @alpha.setter
    def alpha(self, value):
        if value.shape == self.alpha_before_activation.shape:
            self.alpha_before_activation.data = self.alpha_inv_act(value)
        else:
            self.alpha_before_activation = nn.Parameter(self.alpha_inv_act(value.data))

    @color.setter
    def color(self, value):
        if value.shape == self.color_before_activation.shape:
            self.color_before_activation.data = self.color_inv_act(value)
        else:
            self.color = nn.Parameter(self.color_inv_act(value.data))

    @property
    def principal_axis(self):
        return qvec2rotmat_batched(self.qvec)

    @property
    def rotmat(self):
        return qvec2rotmat_batched(self.qvec)

    @property
    def cov(self):
        return qsvec2rotmat_batched(self.qvec, self.svec)

    @property
    def specular(self):
        assert self.pbr and "specular" in self.fields
        return torch.sigmoid(self.specular_before_activation)

    @property
    def is_densifying(self):
        return (
            self.densify_cfg.enabled
            and self.step > self.densify_cfg.warm_up
            and self.step < self.densify_cfg.end
        )

    def initialize(self, initial_values, raw=False):
        # NOTE: actual initialization is done in trainer
        # raw stands for raw values, i.e. not passed through activation
        if "raw" in initial_values:
            raw = initial_values["raw"]
        self.mean = nn.Parameter(initial_values["mean"])
        self.qvec = nn.Parameter(initial_values["qvec"])
        if not raw:
            self.svec_before_activation = nn.Parameter(
                self.svec_inv_act(initial_values["svec"])
            )
            self.color_before_activation = nn.Parameter(
                self.color_inv_act(initial_values["color"])
            )
            self.alpha_before_activation = nn.Parameter(
                self.alpha_inv_act(initial_values["alpha"])
            )
        else:
            self.svec_before_activation = nn.Parameter(initial_values["svec"])
            self.color_before_activation = nn.Parameter(initial_values["color"])
            self.alpha_before_activation = nn.Parameter(initial_values["alpha"])
        self.N = self.mean.data.shape[0]

        if "mask" in initial_values:
            self.grad_mask = initial_values["mask"]
            assert self.grad_mask.shape[0] == self.mean.shape[0]

        if self.pbr:
            self.specular_before_activation = nn.Parameter(
                self.alpha_inv_act(torch.zeros(self.N, 3) + 0.05)
            )
            if self.normal_type == "learned":
                self.normal_before_activation = nn.Parameter(
                    estimate_normal(self.mean, **self.cfg.normal)
                )

    def setup_bg(self, cfg):
        bg_type = cfg.type
        if bg_type == "random":
            self.bg = RandomBackground(cfg)
        elif bg_type == "learned_const":
            self.bg = ConstBackground(cfg)
        elif bg_type == "mlp":
            self.bg = MLPBackground(cfg)
        elif bg_type == "fixed":
            self.bg = FixedBackground(cfg)
        else:
            raise NotImplementedError(f"Background type {bg_type} not implemented")

    def setup(self, cfg):
        # camera imaging params
        self.tile_size = cfg.tile_size
        # frustum culling params
        self.frustum_culling_radius = cfg.frustum_culling_radius

        # tile culling params
        self.tile_culling_type = cfg.tile_culling_type
        self.tile_culling_radius = cfg.tile_culling_radius
        self.tile_culling_thresh = cfg.tile_culling_thresh

        # rendering params
        self.T_thresh = cfg.T_thresh

        # adaptive control params
        self.densify_cfg = cfg.densify
        self.prune_cfg = cfg.prune
        self.densify_enabled = cfg.densify.enabled
        self.register_buffer("max_radii2d", torch.zeros(self.N))
        if self.densify_enabled:
            self.register_buffer("mean_2d_grad_accum", torch.zeros(self.N))
            self.register_buffer("cnt", torch.zeros(self.N))

        # tests
        self.depth_detach = cfg.get("depth_detach", True)

        self.setup_bg(cfg.background)

        self.skip_frustum_culling = cfg.get("skip_frustum_culling", False)

        self.fields = ["mean", "qvec", "svec", "color", "alpha"]
        self.raw_fields = [
            "mean",
            "qvec",
            "svec_before_activation",
            "color_before_activation",
            "alpha_before_activation",
        ]

        if self.cfg.get("pbr", False):
            self.fields.append("specular")
            self.raw_fields.append("specular_before_activation")

            if self.normal_type == "learned":
                self.fields.append("normal")
                self.raw_fields.append("normal_before_activation")

    def setup_lr(self, cfg):
        # should be called in system initialization, not in renderer construction, since this will not saved in the checkpoint
        fields = self.fields + [
            "bg",
        ]
        for field in fields:
            schedule = to_primitive(cfg[field])

            if isinstance(schedule, list):
                if len(schedule) == 4:
                    try:
                        lr_start, lr_end, max_steps, lr_type = schedule
                        scheduler = lr_schedulers[lr_type](max_steps, lr_start, lr_end)
                    except KeyError:
                        scheduler = C_wrapped(schedule)
                elif len(schedule) == 5:
                    scheduler = C_wrapped(schedule)
                else:
                    raise ValueError(f"Invalid lr schedule {schedule}")
            elif isinstance(schedule, float):
                scheduler = lr_schedulers["nothing"](
                    0, schedule, schedule, 0, "nothing"
                )
            else:
                raise ValueError(f"Invalid lr schedule {schedule}")
            setattr(self, f"{field}_lr_scheduler", scheduler)

    def get_params_for_save(self):
        param_groups = {
            "cfg": to_primitive(self.cfg),
            "mean": self.mean.data,
            "qvec": self.qvec.data,
            "svec": self.svec_before_activation.data,
            "color": self.color_before_activation.data,
            "alpha": self.alpha_before_activation.data,
            "bg": self.bg.state_dict(),
        }
        if self.pbr:
            if "specular" in self.fields:
                param_groups["specular"] = self.specular_before_activation.data

            if "normal" in self.fields:
                param_groups["normal"] = self.normal_before_activation.data
        return param_groups

    @classmethod
    def load(cls, cfg, ckpt):
        if cfg is None:
            cfg = {}
        if not isinstance(ckpt, dict):
            ckpt = torch.load(ckpt, map_location="cpu")
        if not "params" in ckpt:
            # case for loading only renderer ckpt
            new_cfg = OmegaConf.create(ckpt["cfg"])
            new_cfg.update(cfg)
            del ckpt["cfg"]
            cfg = new_cfg
        else:
            new_cfg = OmegaConf.create(ckpt["cfg"]).renderer
            new_cfg.update(cfg)
            ckpt = ckpt["params"]
        # if ckpt["color"].max() > 1 or ckpt["color"].min() < 0:
        #     ckpt["color"] = torch.sigmoid(ckpt["color"])
        renderer = cls(new_cfg, ckpt)
        renderer.initialize(ckpt, True)
        try:
            renderer.bg.load_state_dict(ckpt["bg"])
        except:
            console.print(
                "Error when loading background parameters, the background will be randomly initialized"
            )

        return renderer

    def register_mask(self, mask=None):
        if mask is None:
            if hasattr(self, "grad_mask"):
                mask = self.grad_mask
            else:
                mask = torch.ones_like(self.mean[..., 0], dtype=torch.float32)
                console.print("[red]mask assign to all ones since no mask is provided")
                # print(mask.shape)

        self.hooks = {}

        for raw_field in self.raw_fields:
            self.hooks[raw_field] = getattr(self, raw_field).register_hook(
                # lambda grad: grad * mask
                lambda grad: grad
                * (mask if grad.ndim == 1 else mask[..., None])
            )

        console.print("[red]Applied mask to gradients")

    def remove_mask(self):
        for raw_field in self.raw_fields:
            self.hooks[raw_field].remove()

        del self.hooks
        console.print("[red]Removed mask to gradients")

    def get_params_by_mask(self, mask):
        mask = mask.to(self.device)
        params = {
            "mean": self.mean[mask],
            "qvec": self.qvec[mask],
            "svec": self.svec_before_activation[mask],
            "color": self.color_before_activation[mask],
            "alpha": self.alpha_before_activation[mask],
        }
        if self.pbr and "specular" in self.fields:
            params["specular"] = self.specular_before_activation[mask]
        if self.pbr and "normal" in self.fields:
            params["normal"] = self.normal_before_activation[mask]
        return params

    def get_param_groups(self):
        param_groups = {
            "mean": self.mean,
            "qvec": self.qvec,
            "svec": self.svec_before_activation,
            "color": self.color_before_activation,
            "alpha": self.alpha_before_activation,
            "bg": self.bg.parameters(),
        }
        if self.pbr and "specular" in self.fields:
            param_groups["specular"] = self.specular_before_activation
        if self.pbr and "normal" in self.fields:
            param_groups["normal"] = self.normal_before_activation
        return param_groups

    def set_optimizer(self, cfg, step=0):
        lr = 0.0
        opt_params = []

        param_groups = self.get_param_groups()
        for name, params in param_groups.items():
            opt_params.append(
                {
                    "params": params,
                    "lr": getattr(self, f"{name}_lr_scheduler")(step),
                    "name": name,
                }
            )

        opt_cls = getattr(torch.optim, cfg.type)

        self.opt_cfg = cfg
        if cfg.opt_args is None:
            opt_args = {}
        else:
            opt_args = cfg.opt_args
        self.optimizer = opt_cls(opt_params, lr=lr, **opt_args)

    def prune_optimizer(self, mask):
        # NOTE: this is a hacky way to prune the optimizer, but it works (only for Adam)
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "bg":
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                for key in stored_state.keys():
                    if stored_state[key].ndim == 0:
                        continue
                    stored_state[key] = stored_state[key][mask]
                    # stored_state[key] = stored_state[key][mask]
                # stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                # stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def update_lr(self, step):
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            param_group["lr"] = getattr(self, f"{name}_lr_scheduler")(step)

    def update(self, step):
        self.update_lr(step)
        if self.mask_enabled:
            if step == self.mask_start_step:
                self.register_mask()
            elif step == self.mask_end_step:
                self.remove_mask()

    def update_densify_info(self):
        for mean_2d, mask in zip(self.mean_2ds, self.masks):
            self.mean_2d_grad_accum[mask] += mean_2d.grad.data.norm(dim=-1)
            self.cnt[mask] += 1
        self.mean_2ds = []
        self.masks = []

    def update_params_with_dict(self, new_params_dict):
        for field in self.fields:
            setattr(self, field2raw[field], new_params_dict[field])
        self.N = self.mean.data.shape[0]

    def reset_densify_info(self):
        self.mean_2d_grad_accum = torch.zeros_like(self.mean[..., 0])
        self.cnt = torch.zeros_like(self.mean_2d_grad_accum)
        self.max_radii2d = torch.zeros_like(self.mean_2d_grad_accum)

    def densify_on_optimizer(self, new_params_dict):
        # NOTE: only applicable to Adam
        updated_params = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "bg":
                continue
            assert len(group["params"]) == 1, f"{group['name']}"
            extension_tensor = new_params_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                # stored_state["exp_avg"] = torch.cat(
                #     (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                # )
                # stored_state["exp_avg_sq"] = torch.cat(
                #     (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                #     dim=0,
                # )
                for key in stored_state.keys():
                    if stored_state[key].ndim == 0:
                        continue
                    stored_state[key] = torch.cat(
                        (stored_state[key], torch.zeros_like(extension_tensor)), dim=0
                    )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                updated_params[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                updated_params[group["name"]] = group["params"][0]

        return updated_params

    def densify_with_new_params(self, new_params_dict):
        updated_params = self.densify_on_optimizer(new_params_dict)
        self.update_params_with_dict(updated_params)

    def prune_by_mask(self, mask):
        valid_gaussian_mask = ~mask
        pruned_params = self.prune_optimizer(valid_gaussian_mask)
        self.update_params_with_dict(pruned_params)
        self.N = torch.count_nonzero(valid_gaussian_mask).item()
        try:
            self.max_radii2d = self.max_radii2d[valid_gaussian_mask]
        except IndexError:
            self.max_radii2d = torch.zeros(
                self.N, dtype=torch.float32, device=self.device
            )

        try:
            self.mean_2d_grad_accum = self.mean_2d_grad_accum[valid_gaussian_mask]
            self.cnt = self.cnt[valid_gaussian_mask]
        except IndexError:
            self.mean_2d_grad_accum = torch.zeros(
                self.N, dtype=torch.float32, device=self.device
            )
            self.cnt = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        except AttributeError:
            assert not self.densify_cfg.enabled

    def densify_by_split(self, grads, grad_thresh, n_splits=2, mask=None):
        if mask is not None:
            # make the split can be overriden by mask
            selected_pts_mask = mask
        else:
            n_init_points = self.mean.shape[0]
            # Extract points that satisfy the gradient condition
            padded_grad = torch.zeros((n_init_points), device="cuda")
            padded_grad[: grads.shape[0]] = grads.squeeze()
            selected_pts_mask = torch.where(padded_grad >= grad_thresh, True, False)
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(self.svec, dim=1).values > self.densify_cfg.split_thresh,
            )
        new_mean = self.mean.data[selected_pts_mask].repeat(n_splits, 1)
        new_qvec = self.qvec.data[selected_pts_mask].repeat(n_splits, 1)
        new_svec = self.svec.data[selected_pts_mask].repeat(n_splits, 1)
        new_raw_color = self.color_before_activation.data[selected_pts_mask].repeat(
            n_splits, 1
        )
        new_raw_alpha = self.alpha_before_activation.data[selected_pts_mask].repeat(
            n_splits
        )
        split_rotmat = qvec2rotmat_batched(new_qvec).transpose(-1, -2)
        num_splits_gaussians = torch.count_nonzero(selected_pts_mask).item()
        split_gn = (
            torch.randn(num_splits_gaussians * n_splits, 3, device=self.mean.device)
            * new_svec
        )
        new_mean = new_mean + torch.einsum("bij, bj -> bi", split_rotmat, split_gn)
        new_raw_svec = self.svec_inv_act(
            new_svec / (n_splits * self.densify_cfg.split_shrink)
        )

        new_params = {
            "mean": new_mean,
            "qvec": new_qvec,
            "svec": new_raw_svec,
            "color": new_raw_color,
            "alpha": new_raw_alpha,
        }
        if self.pbr and "specular" in self.fields:
            new_params["specular"] = self.specular_before_activation.data[
                selected_pts_mask
            ].repeat(n_splits, 1)
        if self.pbr and "normal" in self.fields:
            new_params["normal"] = self.normal_before_activation.data[
                selected_pts_mask
            ].repeat(n_splits, 1)
        self.densify_with_new_params(new_params)
        prune_mask = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    n_splits * selected_pts_mask.sum(), device="cuda", dtype=bool
                ),
            )
        )
        # print(selected_pts_mask)
        self.prune_by_mask(prune_mask)

        return torch.count_nonzero(selected_pts_mask).item()

    def densify_by_clone(self, grads, grad_thresh, mask=None):
        if mask is None:
            selected_pts_mask = torch.where(
                torch.norm(grads, dim=-1) >= grad_thresh, True, False
            )
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(self.svec, dim=1).values <= self.densify_cfg.split_thresh,
            )
        else:
            selected_pts_mask = mask
        new_params = self.get_params_by_mask(selected_pts_mask)
        self.densify_with_new_params(new_params)

        return torch.count_nonzero(selected_pts_mask).item()

    def densify_by_scale(self):
        mask = (self.svec > self.cfg.densify.scale_max).any(dim=-1)
        return self.densify_by_split(None, None, self.cfg.densify.n_splits, mask)

    def densify_by_compatnes_with_idx(self, idx):
        nn_svec = self.svec[idx]
        nn_rotmat = self.rotmat[idx]
        nn_pos = self.mean[idx]

        nn_gaussian_surface_dist = distance_to_gaussian_surface(
            nn_pos, nn_svec, nn_rotmat, self.mean
        )
        gaussian_surface_dist = distance_to_gaussian_surface(
            self.mean, self.svec, self.rotmat, nn_pos
        )

        dist_to_nn = torch.norm(nn_pos - self.mean, dim=-1)
        mask = (gaussian_surface_dist + nn_gaussian_surface_dist) < dist_to_nn
        new_direction = (nn_pos - self.mean.data) / dist_to_nn[..., None]
        new_mean = (
            self.mean.data
            + new_direction
            * (dist_to_nn + gaussian_surface_dist - nn_gaussian_surface_dist)[..., None]
            / 2.0
        )[mask]
        new_raw_color = self.color_before_activation.data[mask]
        new_qvec = self.qvec.data[mask]
        new_raw_alpha = self.alpha_before_activation.data[mask]
        # print(torch.ones_like(self.svec.data[mask]).shape)
        # print(
        #     (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist)[mask].shape
        # )
        new_raw_svec = self.svec_inv_act(
            torch.ones_like(self.svec.data[mask])
            * (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist)[mask][
                ..., None
            ]
            / 6.0
        )
        new_params = {
            "mean": new_mean,
            "qvec": new_qvec,
            "svec": new_raw_svec,
            "color": new_raw_color,
            "alpha": new_raw_alpha,
        }
        if self.pbr and "specular" in self.fields:
            new_params["specular"] = self.specular_before_activation.data[mask]
        if self.pbr and "normal" in self.fields:
            new_params["normal"] = self.normal_before_activation.data[mask]
        return new_params

    def densify_by_compatness(self, K=1):
        _, idx = K_nearest_neighbors(self.mean, K=K + 1)
        num_densified = 0
        new_params_list = []
        for i in range(K):
            new_params = self.densify_by_compatnes_with_idx(idx[:, i])
            new_params_list.append(new_params)
        new_params = {}
        for key in new_params_list[0].keys():
            new_params[key] = torch.cat([p[key] for p in new_params_list], dim=0)
        num_densified = new_params["mean"].shape[0]
        self.densify_with_new_params(new_params)

        return num_densified
        # nn_svec = self.svec[idx]
        # nn_rotmat = self.rotmat[idx]
        # nn_pos = self.mean[idx]

        # nn_gaussian_surface_dist = distance_to_gaussian_surface(
        #     nn_pos, nn_svec, nn_rotmat, self.mean
        # )
        # gaussian_surface_dist = distance_to_gaussian_surface(
        #     self.mean, self.svec, self.rotmat, nn_pos
        # )

        # dist_to_nn = torch.norm(nn_pos - self.mean, dim=-1)
        # mask = (gaussian_surface_dist + nn_gaussian_surface_dist) < dist_to_nn
        # new_direction = (nn_pos - self.mean.data) / dist_to_nn[..., None]
        # new_mean = (
        #     self.mean.data
        #     + new_direction
        #     * (dist_to_nn + gaussian_surface_dist - nn_gaussian_surface_dist)[..., None]
        #     / 2.0
        # )[mask]
        # new_raw_color = self.color_before_activation.data[mask]
        # new_qvec = self.qvec.data[mask]
        # new_raw_alpha = self.alpha_before_activation.data[mask]
        # # print(torch.ones_like(self.svec.data[mask]).shape)
        # # print(
        # #     (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist)[mask].shape
        # # )
        # new_raw_svec = self.svec_inv_act(
        #     torch.ones_like(self.svec.data[mask])
        #     * (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist)[mask][
        #         ..., None
        #     ]
        #     / 3.0
        # )
        # new_params = {
        #     "mean": new_mean,
        #     "qvec": new_qvec,
        #     "svec": new_raw_svec,
        #     "color": new_raw_color,
        #     "alpha": new_raw_alpha,
        # }
        # self.densify_with_new_params(new_params)

        # return int(torch.sum(mask).item())

    def densify_by_shrink_then_compatness(self, shrink_factor: float, K: int = 3):
        self.svec = self.svec / shrink_factor
        return self.densify_by_compatness(K=K)

    def densify_by_all(self):
        # densify all gaussians
        return self.densify_by_split(
            None, None, 2, torch.ones_like(self.mean[..., 0], dtype=torch.bool)
        )

    def densify(self, step, verbose=True):
        # check if densify is enabled and triggered, if true, do densify
        if not self.densify_cfg.enabled:
            return
        if step < self.densify_cfg.warm_up or step > self.densify_cfg.end:
            return

        if step_check(step, self.densify_cfg.period, True):
            if self.densify_cfg.use_legacy:
                self.densify_legacy(step, verbose)
                if "shrink_then_compatness" in self.densify_cfg.type:
                    self.densify_by_shrink_then_compatness(
                        self.densify_cfg.get("surface_shrink", 1.5),
                        K=self.densify_cfg.get("K", 3),
                    )
                elif "compatness" in self.densify_cfg.type:
                    self.densify_by_compatness(K=self.densify_cfg.get("K", 3))
            else:
                grads = self.mean_2d_grad_accum / self.cnt
                grads[grads.isnan()] = 0.0
                if self.densify_cfg.type == "official":
                    # The order is important due my shit implementation, the grads need to be padded after clone
                    num_clone = self.densify_by_clone(
                        grads, self.densify_cfg.mean2d_thresh
                    )
                    num_split = self.densify_by_split(
                        grads, self.densify_cfg.mean2d_thresh, self.densify_cfg.n_splits
                    )
                    if verbose:
                        console.print(
                            f"Step {step}| {self.N} gaussians remaining ... | num_split: {num_split}; num_clone: {num_clone} | densify type: {self.densify_cfg.type}",
                            style="magenta",
                        )
                elif self.densify_cfg.type == "scale":
                    num_new_gaussians = self.densify_by_scale()
                    if verbose:
                        console.print(
                            f"Step {step}| {self.N} gaussians remaining ... | num_split: {num_new_gaussians} | densify type: {self.densify_cfg.type}",
                            style="magenta",
                        )
                elif self.densify_cfg.type == "compatness":
                    num_new_gaussians = self.densify_by_compatness(
                        K=self.densify_cfg.get("K", 3)
                    )
                    if verbose:
                        console.print(
                            f"Step {step}| {self.N} gaussians remaining ... | num_split: {num_new_gaussians} | densify type: {self.densify_cfg.type}",
                            style="magenta",
                        )
                elif self.densify_cfg.type == "all":
                    num_new_gaussians = self.densify_by_all()
                    if verbose:
                        console.print(
                            f"Step {step}| {self.N} gaussians remaining ... | num_split: {num_new_gaussians} | densify type: {self.densify_cfg.type}",
                            style="magenta",
                        )
                elif self.densify_cfg.type == "shrink_then_compatness":
                    num_new_gaussians = self.densify_by_shrink_then_compatness(
                        self.densify_cfg.get("surface_shrink", 1.5),
                        K=self.densify_cfg.get("K", 3),
                    )
                else:
                    raise NotImplementedError(
                        f"Unknown densify type: {self.densify_cfg.type}"
                    )

            # all methods needs this to reset the accumulators
            self.reset_densify_info()

    def densify_legacy(self, step, verbose=False):
        # TODO: add old adapative control code here from compatibility
        assert (
            self.mean.grad is not None
        ), "mean.grad is None while clone or split gaussians are performed according to spatial gradient of mean"

        # all the gaussians need split or clone
        mask = (
            self.mean_2d_grad_accum / (self.cnt + 1e-5) > self.densify_cfg.mean2d_thresh
        )

        svec_mask = (self.svec.data > self.densify_cfg.split_thresh).any(dim=-1)
        # gaussians need split are with large spatial scale
        split_mask = torch.logical_and(mask, svec_mask)
        # gaussians need clone are with small spatial scale
        clone_mask = torch.logical_and(mask, torch.logical_not(split_mask))

        del mask, svec_mask

        num_split = split_mask.sum().item()
        num_clone = clone_mask.sum().item()

        # number of gaussians after split and clone will be increased by num_split + num_clone
        num_new_gaussians = num_split + num_clone

        # split
        split_mean = self.mean.data[split_mask].repeat(2, 1)
        split_qvec = self.qvec.data[split_mask].repeat(2, 1)
        split_svec = self.svec.data[split_mask].repeat(2, 1)
        # split_svec_ba = self.svec_before_activation.data[split_mask].repeat(2, 1)
        split_color = self.color_before_activation.data[split_mask].repeat(2, 1)
        split_alpha_ba = self.alpha_before_activation.data[split_mask].repeat(2)
        split_rotmat = qvec2rotmat_batched(split_qvec).transpose(-1, -2)
        if self.pbr and "specular" in self.fields:
            split_specular = self.specular_before_activation.data[split_mask].repeat(
                2, 1
            )
        if self.pbr and "normal" in self.fields:
            split_normal = self.normal_before_activation.data[split_mask].repeat(2, 1)

        split_gn = torch.randn(num_split * 2, 3, device=self.mean.device) * split_svec

        split_sampled_mean = split_mean + torch.einsum(
            "bij, bj -> bi", split_rotmat, split_gn
        )

        # check left product or right product

        old_num_gaussians = self.N
        unchanged_gaussians = old_num_gaussians - num_split
        self.N += num_new_gaussians

        new_mean = torch.zeros([self.N, 3], device=self.device)
        new_qvec = torch.zeros([self.N, 4], device=self.device)
        new_svec = torch.zeros([self.N, 3], device=self.device)
        new_color = torch.zeros([self.N, 3], device=self.device)
        new_alpha = torch.zeros([self.N], device=self.device)

        # copy old gaussians (# old_N - num_split)
        new_mean[:unchanged_gaussians] = self.mean.data[~split_mask]
        new_qvec[:unchanged_gaussians] = self.qvec.data[~split_mask]
        new_svec[:unchanged_gaussians] = self.svec_before_activation.data[~split_mask]
        new_color[:unchanged_gaussians] = self.color_before_activation.data[~split_mask]
        new_alpha[:unchanged_gaussians] = self.alpha_before_activation.data[~split_mask]

        # clone gaussians
        new_mean[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.mean.data[clone_mask]
        new_qvec[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.qvec.data[clone_mask]
        new_svec[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.svec_before_activation.data[clone_mask]
        new_color[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.color_before_activation.data[clone_mask]
        new_alpha[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.alpha_before_activation.data[clone_mask]

        pts = unchanged_gaussians + num_clone

        new_mean[pts : pts + 2 * num_split] = split_sampled_mean
        new_qvec[pts : pts + 2 * num_split] = split_qvec
        new_color[pts : pts + 2 * num_split] = split_color
        new_alpha[pts : pts + 2 * num_split] = split_alpha_ba
        new_svec[pts : pts + 2 * num_split] = self.svec_inv_act(
            split_svec / self.densify_cfg.split_shrink / 2.0
        )

        assert pts + 2 * num_split == self.N
        self.mean = torch.nn.Parameter(new_mean)
        self.qvec = torch.nn.Parameter(new_qvec)
        self.svec_before_activation = torch.nn.Parameter(new_svec)
        self.color_before_activation = torch.nn.Parameter(new_color)
        self.alpha_before_activation = torch.nn.Parameter(new_alpha)

        if self.pbr and "specular" in self.fields:
            new_specular = torch.zeros([self.N, 3], device=self.device)
            new_specular[:unchanged_gaussians] = self.specular_before_activation.data[
                ~split_mask
            ]
            new_specular[
                unchanged_gaussians : unchanged_gaussians + num_clone
            ] = self.specular_before_activation.data[clone_mask]
            new_specular[pts : pts + 2 * num_split] = split_specular
            self.specular_before_activation = torch.nn.Parameter(new_specular)

        if self.pbr and "normal" in self.fields:
            new_normal = torch.zeros([self.N, 3], device=self.device)
            new_normal[:unchanged_gaussians] = self.normal_before_activation.data[
                ~split_mask
            ]
            new_normal[
                unchanged_gaussians : unchanged_gaussians + num_clone
            ] = self.normal_before_activation.data[clone_mask]
            new_normal[pts : pts + 2 * num_split] = split_normal
            self.normal_before_activation = torch.nn.Parameter(new_normal)

        self.set_optimizer(self.opt_cfg, step)

        if verbose:
            console.print(
                f"Step {step}| {self.N} gaussians remaining ... | num_split: {num_split}; num_clone: {num_clone} | densify type: legacy",
                style="magenta",
            )

    ### penalty functions
    def alpha_penalty_loss(self, step, writer):
        alpha_penalty_weight = C(self.cfg.penalty.alpha.value, step, None)
        alpha_penalty = 0.0
        if alpha_penalty_weight > 0.0:
            alpha_penalty_type = self.cfg.penalty.alpha.type
            if alpha_penalty_type == "uniform_l1":
                alpha_penalty = torch.mean(self.alpha)
            elif alpha_penalty_type == "uniform_l2":
                alpha_penalty = torch.mean(self.alpha**2)
            elif alpha_penalty_type == "center_weighted":
                alpha_penalty = self.mean.detach().norm(dim=-1) * self.alpha
                alpha_penalty = torch.mean(alpha_penalty)
            else:
                raise ValueError(f"Unknown alpha penalty type: {alpha_penalty_type}")
            writer.add_scalar("auxiliary/alpha_penalty", alpha_penalty.item(), step)
            writer.add_scalar(
                "auxiliary/alpha_penalty_weight", alpha_penalty_weight, step
            )
            return alpha_penalty_weight * alpha_penalty
        else:
            return torch.zeros_like(self.alpha[0], requires_grad=False)

    def mean_penalty_loss(self, step, writer):
        if not hasattr(self.cfg.penalty, "mean"):
            return torch.zeros_like(self.alpha[0], requires_grad=False)
        mean_penalty_weight = C(self.cfg.penalty.mean.value, step, None)
        mean_penalty = 0.0
        if mean_penalty_weight > 0.0:
            mean_penalty_type = self.cfg.penalty.mean.type
            if mean_penalty_type == "uniform_l1":
                mean_penalty = torch.mean(self.mean.norm(dim=-1))
            elif mean_penalty_type == "uniform_l2":
                mean_penalty = torch.mean(self.mean.norm(dim=-1) ** 2)
            elif mean_penalty_type == "weighted_l1":
                mean_penalty = torch.mean(
                    self.mean.norm(dim=-1).detach() * self.mean.norm(dim=-1)
                )
            elif mean_penalty_type == "weighted_l2":
                mean_penalty = torch.mean(
                    self.mean.norm(dim=-1).detach() ** 2 * self.mean.norm(dim=-1) ** 2
                )
            else:
                raise ValueError(f"Unknown mean penalty type: {mean_penalty_type}")
            writer.add_scalar("auxiliary/mean_penalty", mean_penalty.item(), step)
            writer.add_scalar(
                "auxiliary/mean_penalty_weight", mean_penalty_weight, step
            )
            return mean_penalty_weight * mean_penalty
        else:
            return torch.zeros_like(self.alpha[0], requires_grad=False)

    def scale_penalty_loss(self, step, writer):
        scale_penalty_weight = C(self.cfg.penalty.scale.value, step, None)
        scale_penalty = 0.0
        if scale_penalty_weight:
            svec = self.svec
            volume = svec.prod(dim=-1).sum()
            writer.add_scalar("auxiliary/scale_penalty", volume.item(), step)
            writer.add_scalar(
                "auxiliary/scale_penalty_weight", scale_penalty_weight, step
            )
            return scale_penalty_weight * volume
        else:
            return torch.zeros_like(self.alpha[0], requires_grad=False)

    def move_penalty_loss(self, step, writer):
        ## move penalty: penalize the movement of the gaussian, i.e. the distance between the mean of the gaussian and the mean of the gaussian in the previous frame should be small
        if not hasattr(self.cfg.penalty, "move"):
            return torch.zeros_like(self.alpha[0], requires_grad=False)
        move_penalty_weight = C(self.cfg.penalty.move.value, step, None)
        if move_penalty_weight > 0.0:
            move_penalty = torch.mean(
                (self.mean - self.prev_mean.detach()).norm(dim=-1)
            )
            writer.add_scalar("auxiliary/move_penalty", move_penalty.item(), step)
            writer.add_scalar(
                "auxiliary/move_penalty_weight", move_penalty_weight, step
            )
            return move_penalty_weight * move_penalty
        else:
            return torch.zeros_like(self.alpha[0], requires_grad=False)

    def NN_penalty_loss(self, step, writer):
        ## nearest neighbor penalty: penalize the distance between the mean of the gaussian and the mean of the nearest gaussian
        if not hasattr(self.cfg.penalty, "NN"):
            return torch.zeros_like(self.alpha[0], requires_grad=False)
        NN_penalty_weight = C(self.cfg.penalty.NN.value, step, None)
        if NN_penalty_weight > 0.0:
            # TODO: find a good implemenation of KNN
            NN, _ = nearest_neighbor(self.mean)
            NN_penalty = torch.mean((self.mean - NN).norm(dim=-1))
            writer.add_scalar("auxiliary/NN_penalty", NN_penalty.item(), step)
            writer.add_scalar("auxiliary/NN_penalty_weight", NN_penalty_weight, step)
            return NN_penalty_weight * NN_penalty
        else:
            return torch.zeros_like(self.alpha[0], requires_grad=False)

    @lineprofiler
    def compat_penalty_loss(self, step, writer):
        # TODO: finish this
        compat_penalty_weight = C(self.cfg.penalty.compat.value, step, None)
        if compat_penalty_weight > 0.0:
            compat_penalty = 0.0

            _, idx = nearest_neighbor(self.mean)
            nn_svec = self.svec[idx]
            nn_rotmat = self.rotmat[idx]
            nn_pos = self.mean[idx]

            nn_gaussian_surface_dist = distance_to_gaussian_surface(
                nn_pos, nn_svec, nn_rotmat, self.mean
            )
            gaussian_surface_dist = distance_to_gaussian_surface(
                self.mean, self.svec, self.rotmat, nn_pos
            )

            dist_to_nn = torch.norm(nn_pos - self.mean, dim=-1)
            mask = (gaussian_surface_dist + nn_gaussian_surface_dist) < dist_to_nn
            # console.print(f"effective ones: {torch.sum(mask).item()}/{self.N}")

            if self.cfg.penalty.compat.type == "l1":
                compat_penalty = torch.mean(
                    (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist)
                    * mask
                )
            elif self.cfg.penalty.compat.type == "l2":
                compat_penalty = torch.mean(
                    (dist_to_nn - gaussian_surface_dist - nn_gaussian_surface_dist) ** 2
                    * mask
                )
            else:
                raise ValueError(
                    f"Unknown compat penalty type: {self.cfg.penalty.compat.type}"
                )

            writer.add_scalar("auxiliary/compat_penalty", compat_penalty.item(), step)
            writer.add_scalar(
                "auxiliary/effective_rate", torch.sum(mask).item() / self.N, step
            )
            writer.add_scalar(
                "auxiliary/compat_penalty_weight", compat_penalty_weight, step
            )
            return compat_penalty_weight * compat_penalty
        else:
            return torch.zeros_like(self.alpha[0], requires_grad=False)

    def normal_penalty_loss(self, step, writer):
        pass

    def specular_penalty_loss(self, step, writer):
        if not hasattr(self.cfg.penalty, "specular"):
            return torch.zeros_like(self.alpha[0], requires_grad=False)
        specular_penalty_weight = C(self.cfg.penalty.move.value, step, None)
        if specular_penalty_weight > 0.0:
            specular_penalty = torch.mean(self.specular)
            writer.add_scalar(
                "auxiliary/specular_penalty", specular_penalty.item(), step
            )
            writer.add_scalar(
                "auxiliary/specular_penalty_weight", specular_penalty_weight, step
            )
            return specular_penalty_weight * specular_penalty
        else:
            return torch.zeros_like(self.alpha[0], requires_grad=False)

    def auxiliary_loss(self, step, writer):
        loss = 0.0
        for key in self.cfg.penalty:
            loss += getattr(self, f"{key}_penalty_loss")(step, writer)
        writer.add_scalar("auxiliary/total", loss.item(), step)

        return loss

    ### prune
    def prune_by_scale(self, step):
        # prune large scales
        assert hasattr(self, "max_radii2d"), "max_radii2d not set"
        radii2d_thresh = C(self.prune_cfg.radii2d_thresh, step, None)
        mask = self.max_radii2d > radii2d_thresh
        self.prune_by_mask(mask)
        num_pruned = torch.sum(mask).item()
        self.N = self.mean.shape[0]
        return num_pruned

    def prune_by_alpha(self, step):
        alpha_thresh = C(self.prune_cfg.alpha_thresh, step, None)
        mask = self.alpha < alpha_thresh
        self.prune_by_mask(mask)
        num_pruned = torch.sum(mask).item()
        self.N = self.mean.shape[0]
        return num_pruned

    def prune_by_svec(self, step):
        if not hasattr(self.prune_cfg, "radii3d_thresh"):
            return 0
        svec_thresh = self.prune_cfg.radii3d_thresh
        mask = (self.svec > svec_thresh).all(dim=-1)
        self.prune_by_mask(mask)
        num_pruned = torch.sum(mask).item()
        self.N = self.mean.shape[0]
        return num_pruned

    def prune(self, step, verbose=True):
        if not self.prune_cfg.enabled:
            return
        if step < self.prune_cfg.warm_up or step > self.prune_cfg.end:
            return
        if step_check(step, self.prune_cfg.period):
            if self.prune_cfg.radii2d_thresh > 0.0:
                num_pruned_by_scale = self.prune_by_scale(step)
            else:
                num_pruned_by_scale = 0
            if self.prune_cfg.alpha_thresh > 0.0:
                num_pruned_by_alpha = self.prune_by_alpha(step)
            else:
                num_pruned_by_alpha = 0
            if (
                hasattr(self.prune_cfg, "radii3d_thresh")
                and self.prune_cfg.radii3d_thresh > 0.0
            ):
                num_pruned_by_svec = self.prune_by_svec(step)
            else:
                num_pruned_by_svec = 0
            if verbose:
                console.print(
                    f"Step {step}| {self.N} gaussians remaining ... | pruned by scale: {num_pruned_by_scale}| pruned by alpha: {num_pruned_by_alpha}| prune by radii3d {num_pruned_by_svec}",
                    style="magenta",
                )

    def get_with_overrides(self, field, overrides):
        if overrides is None or field not in overrides:
            return getattr(self, field)
        else:
            return overrides[field]

    @lineprofiler
    def update_normal(self):
        self.normal = estimate_normal(self.mean, **self.cfg.normal)

    @property
    def normal(self):
        if self.normal_type == "estimated":
            return estimate_normal(self.mean, **self.cfg.normal)
        elif self.normal_type == "learned":
            return F.normalize(torch.tanh(self.normal_before_activation), dim=-1)
        else:
            raise ValueError(f"Unknown normal type: {self.normal_type}")

    def render_one(
        self,
        c2w,
        camera_info,
        use_bg=True,
        rgb_only=False,
        overrides=None,
        return_T=False,
    ):
        # overrides is a dict of tensors that will override the corresponding
        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        if self.skip_frustum_culling:
            mask = torch.ones_like(mask, dtype=torch.bool, device=self.device)
        else:
            with torch.no_grad():
                _backend.culling_gaussian_bsphere(
                    self.mean,
                    self.qvec,
                    self.svec,
                    f_normals,
                    f_pts,
                    mask,
                    self.frustum_culling_radius,
                )

        mean = self.get_with_overrides("mean", overrides)
        qvec = self.get_with_overrides("qvec", overrides)
        svec = self.get_with_overrides("svec", overrides)
        color = self.get_with_overrides("color", overrides)
        alpha = self.get_with_overrides("alpha", overrides)

        mean = mean[mask].contiguous()
        qvec = qvec[mask].contiguous()
        svec = svec[mask].contiguous()
        color = color[mask].contiguous()
        alpha = alpha[mask].contiguous()

        mean, cov, JW, depth = project_gaussians(
            mean, qvec, svec, c2w, self.depth_detach
        )

        if self.training:
            with torch.no_grad():
                m = (cov[..., 0, 0] + cov[..., 1, 1]) / 2.0
                p = torch.det(cov)
                radii2d = m + torch.sqrt((m**2 - p).clamp(min=0))
                self.max_radii2d[mask] = torch.max(self.max_radii2d[mask], radii2d)
        if self.training and self.densify_enabled:
            mean_2d = mean
            mean_2d.retain_grad()
            self.mean_2ds.append(mean_2d)
            self.masks.append(mask)
            # ??
            # with torch.no_grad():
            #     m = (cov[..., 0, 0] + cov[..., 1, 1]) / 2.0
            #     p = torch.det(cov)
            #     radii2d = m + torch.sqrt(m**2 - p)
            #     self.max_radii2d[mask] = torch.max(self.max_radii2d[mask], radii2d)

        tic()
        N_with_dub, aabb_topleft, aabb_bottomright = tile_culling_aabb_count(
            mean,
            cov,
            self.tile_size,
            camera_info,
            self.tile_culling_radius,
        )
        toc("count N with dub")

        self.total_dub_gaussians = N_with_dub

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)
        start = -torch.ones([n_tiles], dtype=torch.int32, device=self.device)
        end = -torch.ones([n_tiles], dtype=torch.int32, device=self.device)
        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy
        gaussian_ids = torch.zeros([N_with_dub], dtype=torch.int32, device=self.device)

        tic()
        _backend.tile_culling_aabb_start_end(
            aabb_topleft,
            aabb_bottomright,
            gaussian_ids,
            start,
            end,
            depth,
            n_tiles_h,
            n_tiles_w,
        )
        toc("tile culling aabb")

        if self.cfg.debug:
            print_info(end - start, "num_gaussians_per_tile")

        tic()
        # torch.cuda.profiler.cudart().cudaProfilerStart()
        # NOTE: initial value of T is 1.0, which means no occlusion
        T = torch.ones([H, W, 1], device=self.device, dtype=torch.float32)
        rays_d = camera_info.get_rays_d(c2w)
        out = render_with_T(
            mean,
            cov,
            color,
            alpha,
            start,
            end,
            gaussian_ids,
            img_topleft,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            self.T_thresh,
            self.bg(rays_d),
        ).view(H, W, 3)

        # TODO: deprecate this flag
        # if use_bg:
        #     rays_d = camera_info.get_rays_d(c2w)
        #     out = out + T * self.bg(rays_d)

        toc("render with T")

        outputs = {}
        outputs["rgb"] = out

        if not rgb_only:
            tic()
            # scalar = torch.ones([H, W, 1], dtype=torch.float32, device=self.device)
            rendered_depth = render_scalar(
                mean,
                cov,
                depth,
                alpha,
                start,
                end,
                gaussian_ids,
                img_topleft,
                self.tile_size,
                n_tiles_h,
                n_tiles_w,
                pixel_size_x,
                pixel_size_y,
                H,
                W,
                self.T_thresh,
                T,
            ).reshape(H, W, 1)
            toc("render depth")

            tic()
            scalar = torch.ones_like(self.mean.data[..., 0])
            opacity = render_scalar(
                mean,
                cov,
                scalar,
                alpha,
                start,
                end,
                gaussian_ids,
                img_topleft,
                self.tile_size,
                n_tiles_h,
                n_tiles_w,
                pixel_size_x,
                pixel_size_y,
                H,
                W,
                self.T_thresh,
                T,
            ).reshape(H, W, 1)
            toc("render opacity")

            tic()
            z2 = depth * depth
            z2 = render_scalar(
                mean,
                cov,
                z2,
                alpha,
                start,
                end,
                gaussian_ids,
                img_topleft,
                self.tile_size,
                n_tiles_h,
                n_tiles_w,
                pixel_size_x,
                pixel_size_y,
                H,
                W,
                self.T_thresh,
                T,
            ).reshape(H, W, 1)

            z_var = z2 - rendered_depth * rendered_depth
            # TODO: add z var rendering code, z_var = E[z^2] - E[z]^2
            toc("render z var, proposed in HiFA")

            if not self.training:
                out = out.clamp(0, 1)

            outputs.update(
                {
                    "depth": rendered_depth,
                    "opacity": opacity,
                    "z_var": z_var,
                }
            )

            if return_T:
                outputs["T"] = T

        return outputs

    def forward(self, batch, use_bg=True, rgb_only=False):
        if self.cfg.normal_as_rgb:
            # this will set correct normal vector
            self.update_normal()
            overrides = {"color": (self.normal + 1.0) * 0.5}
        elif self.pbr and "specular" in self.fields:
            self.update_normal()
            batch["light_pos"] = batch["light_pos"].to(self.device)
            batch["light_color"] = batch["light_color"].to(self.device)
        else:
            overrides = {}
        c2ws = batch["c2w"].to(self.device)
        bs = c2ws.shape[0]
        camera_infos = batch["camera_info"]

        outputs = []
        for i in range(bs):
            if self.pbr and "specular" in self.fields:
                # breakpoint()
                shaded_color = compute_shaded_color(
                    batch["light_pos"][i],
                    batch["light_color"][i],
                    self.normal,
                    self.specular,
                    self.mean,
                    c2ws[i][:3, 3],
                )
                # breakpoint()
                overrides = {"color": shaded_color + self.color}
            outputs.append(
                self.render_one(
                    c2ws[i],
                    camera_infos[i],
                    use_bg,
                    rgb_only,
                    overrides=overrides,
                )
            )

        outputs = stack_dicts(outputs)
        if self.cfg.normal_as_rgb:
            outputs["rgb"] = (F.normalize(outputs["rgb"], dim=-1, eps=1e-6) + 1.0) / 2.0

        return outputs

    def post_backward(self):
        # call after backward
        if self.training:
            self.update_densify_info()
            self.mean_2ds = []
            self.masks = []

    ## log
    @torch.no_grad()
    def log(self, writer, step):
        self.log_bounds(writer, step)
        self.log_grad_bounds(writer, step)
        self.log_statistics(writer, step)
        self.log_lr(writer, step)
        try:
            writer.add_histogram("hists/max_radii2d", self.max_radii2d, step)
        except:
            warnings.warn("trying to log max_radii2d but it is not set")
        if self.is_densifying:
            self.log_densify_info(writer, step)

    @torch.no_grad()
    def log_bounds(self, writer, step):
        """log the bounds of the parameters"""
        writer.add_scalar("renderer/num_gaussians", self.mean.data.shape[0], step)
        writer.add_scalar(
            "renderer/n_gaussians_with_dub", self.total_dub_gaussians, step
        )
        for field in self.fields:
            if field == "bg":
                continue
            writer.add_scalar(
                f"renderer/{field}/min",
                getattr(self, field).abs().min(),
                step,
            )
            writer.add_scalar(
                f"renderer/{field}/max",
                getattr(self, field).abs().max(),
                step,
            )
            writer.add_scalar(
                f"renderer/{field}/mean",
                getattr(self, field).mean(),
            )

    @torch.no_grad()
    def log_grad_bounds(self, writer, step):
        if self.mean.grad is None:
            return
        for field, raw_field in zip(self.fields, self.raw_fields):
            if field == "bg":
                continue
            try:
                writer.add_scalar(
                    f"renderer/{field}/grad_min",
                    getattr(self, raw_field).grad.abs().min(),
                    step,
                )
                writer.add_scalar(
                    f"renderer/{field}/grad_max",
                    getattr(self, raw_field).grad.abs().max(),
                    step,
                )
            except AttributeError:
                pass

    @torch.no_grad()
    def log_statistics(self, writer, step):
        writer.add_histogram("hists/mean", self.mean.norm(dim=-1).cpu().numpy(), step)
        writer.add_histogram(
            "hists/svec_min", self.svec.min(dim=-1)[0].cpu().numpy(), step
        )
        writer.add_histogram(
            "hists/svec_max", self.svec.max(dim=-1)[0].cpu().numpy(), step
        )
        writer.add_histogram("hists/alpha", self.alpha.cpu().numpy(), step)
        if self.mean.grad is not None:
            # TODO: maybe add more info here
            writer.add_histogram(
                "hists/grad_mean", self.mean.grad.norm(dim=-1).cpu().numpy(), step
            )

    def log_lr(self, writer, step):
        lrs = {}
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            lrs[name] = param_group["lr"]
        for name, value in lrs.items():
            writer.add_scalar(f"lr/{name}", value, step)

    @torch.no_grad()
    def log_densify_info(self, writer, step):
        # TODO
        writer.add_histogram("hists/grad_mean2d", self.mean_2d_grad_accum, step)
        writer.add_histogram("hists/cnt", self.cnt, step)
        writer.add_histogram("hists/max_radii2d", self.max_radii2d, step)

    ## export
    def get_density_val_grid(self, L, reso):
        x = torch.linspace(-L, L, reso)
        y = torch.linspace(-L, L, reso)
        z = torch.linspace(-L, L, reso)

        x, y, z = torch.meshgrid(x, y, z)
        grid = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)
