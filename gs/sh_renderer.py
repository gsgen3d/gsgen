import gc
from typing import Any
import cv2
from pathlib import Path
from utils.camera import PerspectiveCameras
from utils.transforms import qsvec2rotmat_batched, qvec2rotmat_batched
from utils.misc import print_info, tic, toc
import numpy as np
from torchtyping import TensorType
import torch
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from rich.console import Console
from utils.activations import activations, inv_activations
from gs.culling import tile_culling_aabb_count
from utils.misc import lineprofiler
from timeit import timeit
from time import time

# from .initialize import cov_init, alpha_center_anealing_init, alpha_trunc_init
from utils.schedulers import lr_schedulers
from utils.optimizers import optimizers

console = Console()

try:
    import _gs as _backend
except ImportError:
    from .backend import _backend

from gs.renderer import render_sh, step_check, project_gaussians, render_sh_bg


sh_base = 0.28209479177387814
# sh_base = 1.0


@torch.no_grad()
def init_sh_coeffs(cfg, rgb: TensorType["N, 3"], C: int):
    sh_coeffs = torch.zeros((rgb.shape[0], 3, C * C), device=rgb.device)
    sh_coeffs[:, :, 0] = inv_activations["sigmoid"](rgb) / sh_base

    return sh_coeffs


class SHRenderer(torch.nn.Module):
    def __init__(self, cfg, pts=None, rgb=None):
        super().__init__()
        self.device = cfg.device
        self.cfg = cfg
        self.max_C = cfg.sh_order

        self.svec_act = activations[cfg.svec_act]
        self.alpha_act = activations[cfg.alpha_act]

        self.svec_inv_act = inv_activations[cfg.svec_act]
        self.alpha_inv_act = inv_activations[cfg.alpha_act]

        if pts is not None and rgb is not None:
            # self.N = pts.shape[0]
            # self.mean = torch.nn.parameter.Parameter(pts)
            # self.qvec = torch.nn.Parameter(torch.zeros([self.N, 4]))
            # self.qvec.data[..., 0] = 1.0
            # self.sh_coeffs = torch.nn.Parameter(init_sh_coeffs(cfg, rgb, self.max_C))
            # self.svec_before_activation = torch.nn.Parameter(torch.ones([self.N, 3]))
            # self.alpha_before_activation = torch.nn.Parameter(torch.ones([self.N]))
            # self.svec_before_activation.data.fill_(self.svec_inv_act(cfg.svec_init))
            # self.alpha_before_activation.data.fill_(self.alpha_inv_act(cfg.alpha_init))
            self.initialize(cfg, pts, rgb)

        self.now_C = 1
        self.N_changed = False

        if hasattr(self, "mean"):
            self.register_buffer("grad_mean", torch.zeros_like(self.mean[..., 0]))
            self.register_buffer(
                "cnt", torch.zeros_like(self.mean[..., 0], dtype=torch.int32)
            )
        self.set_cfg(cfg)
        self.set_scheduler(cfg)

    def initialize(self, cfg, pts, rgb):
        self.N = pts.shape[0]
        self.mean = torch.nn.parameter.Parameter(pts)
        self.qvec = torch.nn.Parameter(torch.zeros([self.N, 4]))
        self.qvec.data[..., 0] = 1.0
        self.sh_coeffs = torch.nn.Parameter(init_sh_coeffs(cfg, rgb, self.max_C))

        svec_init_method = cfg.get("svec_init_method", "nearest")
        if svec_init_method == "fixed":
            self.svec_before_activation = torch.nn.Parameter(torch.ones([self.N, 3]))
            self.svec_before_activation.data.fill_(self.svec_inv_act(cfg.svec_init))
        elif svec_init_method == "nearest":
            init_svec = (cov_init(pts, cfg.get("nearest_k", 3)) * 10).clamp(
                max=0.1, min=0.01
            )
            print_info(init_svec, "init_svec")
            self.svec_before_activation = torch.nn.Parameter(
                self.svec_inv_act(init_svec).unsqueeze(1).repeat(1, 3)
            )
        elif svec_init_method == "center":
            ## TODO: big alpha for gaussians near the center
            pass
        else:
            raise NotImplementedError

        alpha_init_method = cfg.get("alpha_init_method", "fixed")
        if alpha_init_method == "fixed":
            self.alpha_before_activation = torch.nn.Parameter(torch.ones([self.N]))
            self.alpha_before_activation.data.fill_(self.alpha_inv_act(cfg.alpha_init))
        elif alpha_init_method == "center":
            self.alpha_before_activation = torch.nn.Parameter(torch.ones([self.N]))
            self.alpha_before_activation.data = self.alpha_inv_act(
                alpha_center_anealing_init(pts, None, cfg.alpha_init)
            )
        elif alpha_init_method == "trunc":
            self.alpha_before_activation = torch.nn.Parameter(torch.ones([self.N]))
            self.alpha_before_activation.data = self.alpha_inv_act(
                alpha_trunc_init(pts, cfg.alpha_init, 0.02, cfg.init_pts_bounds)
            )
        else:
            raise NotImplementedError

    def set_cfg(self, cfg):
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
        self.warm_up = cfg.get("warm_up", 1000)
        self.adaptive_control_iteration = cfg.adaptive_control_iteration
        self.pos_grad_thresh = cfg.pos_grad_thresh
        self.split_scale_thresh = cfg.split_scale_thresh
        self.scale_shrink_factor = cfg.scale_shrink_factor
        # !! deprecated
        self.alpha_reset_period = cfg.alpha_reset_period
        self.remove_low_alpha_period = cfg.remove_low_alpha_period
        self.alpha_reset_val = cfg.alpha_reset_val
        self.alpha_thresh = cfg.alpha_thresh

        self.remove_tiny_period = cfg.get("remove_tiny_period", 500)
        self.remove_tiny = cfg.get("remove_tiny", False)
        self.split_type = cfg.get("split_type", "mean_grad")
        self.split_reduction = cfg.get("split_reduction", "max")
        self.svec_thresh = cfg.get("svec_thresh", 50)

        self.remove_large_period = cfg.get("remove_large_period", 500)
        self.world_large_thresh = cfg.get("world_large_thresh", 30)

        # SH
        self.sh_upgrades = cfg.sh_upgrades

        # tests
        self.depth_detach = cfg.get("depth_detach", True)

        self.bg = cfg.get("bg", False)
        if self.bg:
            self.bg_rgb = cfg.get("bg_rgb", [1.0, 1.0, 1.0])
            self.bg_rgb = torch.FloatTensor(self.bg_rgb).to(self.device)

        self.skip_frustum_culling = cfg.get("skip_frustum_culling", False)

        self.fields = ["mean", "qvec", "svec", "sh_coeffs", "alpha"]
        self.opt_type = cfg.get("opt", "adam")
        try:
            self.opt_cls = optimizers[self.opt_type]
            self.opt_args = self.cfg.get("opt_args", dict())
        except KeyError:
            console.print(f"[red bold]Optimizer {self.opt_type} not found")

    def set_bg_rgb(self, rgb):
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.FloatTensor(rgb).to(self.device)
        self.bg_rgb = rgb

    def set_scheduler(self, cfg):
        self.scheduler = dict()
        self.scheduler["mean"] = lr_schedulers[cfg.get("mean_scheduler", "nothing")](
            cfg.max_iteration,
            cfg.mean_lr,
            cfg.get("mean_lr_end", cfg.mean_lr),
            cfg.get("mean_warmup_steps", cfg.warmup_steps),
        )
        self.scheduler["svec"] = lr_schedulers[cfg.get("svec_scheduler", "nothing")](
            cfg.max_iteration,
            cfg.svec_lr,
            cfg.get("svec_lr_end", cfg.svec_lr),
            cfg.get("svec_warmup_steps", cfg.warmup_steps),
        )
        self.scheduler["qvec"] = lr_schedulers[cfg.get("qvec_scheduler", "nothing")](
            cfg.max_iteration,
            cfg.qvec_lr,
            cfg.get("qvec_lr_end", cfg.qvec_lr),
            cfg.get("qvec_warmup_steps", cfg.warmup_steps),
        )
        self.scheduler["sh_coeffs"] = lr_schedulers[
            cfg.get("sh_coeffs_scheduler", "nothing")
        ](
            cfg.max_iteration,
            cfg.sh_coeffs_lr,
            cfg.get("sh_coeffs_lr_end", cfg.sh_coeffs_lr),
            cfg.get("sh_coeffs_warmup_steps", cfg.warmup_steps),
        )
        self.scheduler["alpha"] = lr_schedulers[cfg.get("alpha_scheduler", "nothing")](
            cfg.max_iteration,
            cfg.alpha_lr,
            cfg.get("alpha_lr_end", cfg.alpha_lr),
            cfg.get("alpha_warmup_steps", cfg.warmup_steps),
        )

    def get_with_overrides(self, field, overrides):
        if overrides is None or field not in overrides:
            return getattr(self, field)
        else:
            return overrides[field]

    def forward(self, c2w, camera_info, overrides=None):
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
        sh_coeffs = self.get_with_overrides("sh_coeffs", overrides)
        alpha = self.get_with_overrides("alpha", overrides)

        mean = mean[mask].contiguous()
        qvec = qvec[mask].contiguous()
        svec = svec[mask].contiguous()
        sh_coeffs = sh_coeffs[mask].contiguous()
        alpha = alpha[mask].contiguous()

        mean, cov, JW, depth = project_gaussians(
            mean, qvec, svec, c2w, self.depth_detach
        )

        if hasattr(self, "cnt"):
            self.cnt[mask] += 1

        if self.split_type == "2d_mean_grad" and self.training:
            self.frustum_culling_mask = mask
            self.mean_2d = mean
            self.mean_2d.retain_grad()

        self.depth = depth
        self.radius = None

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
        if not self.bg:
            out = render_sh(
                mean,
                cov,
                sh_coeffs[..., : self.now_C * self.now_C].contiguous(),
                alpha,
                start,
                end,
                gaussian_ids,
                img_topleft,
                c2w,
                self.tile_size,
                n_tiles_h,
                n_tiles_w,
                pixel_size_x,
                pixel_size_y,
                H,
                W,
                self.now_C,
                self.T_thresh,
            )
        else:
            assert hasattr(self, "bg_rgb")
            out = render_sh_bg(
                mean,
                cov,
                sh_coeffs[..., : self.now_C * self.now_C].contiguous(),
                alpha,
                start,
                end,
                gaussian_ids,
                img_topleft,
                c2w,
                self.tile_size,
                n_tiles_h,
                n_tiles_w,
                pixel_size_x,
                pixel_size_y,
                H,
                W,
                self.now_C,
                self.T_thresh,
                self.bg_rgb,
            )
        # torch.cuda.profiler.cudart().cudaProfilerStop()
        toc("render sh")

        return out.reshape(H, W, 3)

    @property
    def svec(self):
        return self.svec_act(self.svec_before_activation)

    @property
    def alpha(self):
        return self.alpha_act(self.alpha_before_activation)

    @property
    def color(self):
        return torch.sigmoid(self.sh_coeffs[..., 0] * sh_base)

    @torch.no_grad()
    def log(self, writer, step):
        self.log_depth_and_radius(writer, step)
        self.log_bounds(writer, step)
        self.log_info(writer, step)
        self.log_grad_bounds(writer, step)
        self.log_n_gaussian_dub(writer, step)
        self.log_statistics(writer, step)
        self.log_lr(writer, step)

    @torch.no_grad()
    def log_lr(self, writer, step):
        params = self.get_param_groups()
        for name in params.keys():
            writer.add_scalar(f"lr/{name}", self.scheduler[name](step), step)

    @torch.no_grad()
    def log_depth_and_radius(self, writer, step):
        """log the bounds of the parameters"""
        if self.depth is not None:
            writer.add_scalar("bounds/depth_max", self.depth.max(), step)
            writer.add_scalar("bounds/depth_min", self.depth.min(), step)
            writer.add_scalar("bounds/depth_mean", self.depth.mean(), step)

        if self.radius is not None:
            writer.add_scalar("bounds/radius_mean", self.radius.mean(), step)
            writer.add_scalar("bounds/radius_max", self.radius.max(), step)
            writer.add_scalar("bounds/radius_min", self.radius.min(), step)

    @torch.no_grad()
    def log_bounds(self, writer, step):
        """log the bounds of the parameters"""
        writer.add_scalar("bounds/mean_max", self.mean.abs().max(), step)
        writer.add_scalar("bounds/mean_min", self.mean.abs().min(), step)
        writer.add_scalar("bounds/qvec_max", self.qvec.abs().max(), step)
        writer.add_scalar("bounds/qvec_min", self.qvec.abs().min(), step)
        writer.add_scalar("bounds/svec_max", self.svec.abs().max(), step)
        writer.add_scalar("bounds/svec_min", self.svec.abs().min(), step)
        writer.add_scalar(
            "bounds/color_max",
            self.sh_coeffs[..., : self.now_C * self.now_C].max(),
            step,
        )
        writer.add_scalar(
            "bounds/color_min",
            self.sh_coeffs[..., : self.now_C * self.now_C].min(),
            step,
        )
        writer.add_scalar("bounds/alpha_max", self.alpha.max(), step)
        writer.add_scalar("bounds/alpha_min", self.alpha.min(), step)

    @torch.no_grad()
    def log_info(self, writer, step):
        writer.add_scalar("info/mean_mean", self.mean.abs().mean(), step)
        writer.add_scalar("info/qvec_mean", self.qvec.abs().mean(), step)
        writer.add_scalar("info/svec_mean", self.svec.abs().mean(), step)
        writer.add_scalar(
            "info/sh_coeffs_mean",
            self.sh_coeffs[..., : self.now_C * self.now_C].abs().mean(),
            step,
        )
        writer.add_scalar("info/alpha_mean", self.alpha.sigmoid().mean(), step)

    @torch.no_grad()
    def log_grad_bounds(self, writer, step):
        if self.mean.grad is None:
            return
        writer.add_scalar("grad_bounds/mean_max", self.mean.grad.abs().max(), step)
        writer.add_scalar("grad_bounds/mean_min", self.mean.grad.abs().min(), step)
        writer.add_scalar("grad_bounds/qvec_max", self.qvec.grad.abs().max(), step)
        writer.add_scalar("grad_bounds/qvec_min", self.qvec.grad.abs().min(), step)
        writer.add_scalar(
            "grad_bounds/svec_max", self.svec_before_activation.grad.abs().max(), step
        )
        writer.add_scalar(
            "grad_bounds/svec_min", self.svec_before_activation.grad.abs().min(), step
        )
        writer.add_scalar(
            "grad_bounds/sh_coeffs_max",
            self.sh_coeffs.grad[..., : self.now_C * self.now_C].max(),
            step,
        )
        writer.add_scalar(
            "grad_bounds/sh_coeffs_min",
            self.sh_coeffs.grad[..., : self.now_C * self.now_C].min(),
            step,
        )
        writer.add_scalar(
            "grad_bounds/alpha_max", self.alpha_before_activation.grad.max(), step
        )
        writer.add_scalar(
            "grad_bounds/alpha_min", self.alpha_before_activation.grad.min(), step
        )

    def log_n_gaussian_dub(self, writer, step):
        writer.add_scalar("n_gaussian_dub", self.total_dub_gaussians, step)

    def split_gaussians_by_radius(self):
        pass

    def split_gaussians(self):
        assert (
            self.mean.grad is not None
        ), "mean.grad is None while clone or split gaussians are performed according to spatial gradient of mean"
        console.print("[red bold]Splitting Gaussians")

        # all the gaussians need split or clone
        if self.split_type == "mean_grad":
            if self.split_reduction == "mean":
                mask = self.grad_mean / (self.cnt + 1e-5) > self.pos_grad_thresh
            elif self.split_reduction == "max":
                mask = self.grad_mean > self.pos_grad_thresh
            else:
                raise NotImplementedError
        elif self.split_type == "2d_mean_grad":
            if self.split_reduction == "mean":
                mask = self.grad_mean / (self.cnt + 1e-5) > self.pos_grad_thresh
            elif self.split_reduction == "max":
                mask = self.grad_mean > self.pos_grad_thresh
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        svec_mask = (self.svec.data > self.split_scale_thresh).any(dim=-1)
        # gaussians need split are with large spatial scale
        split_mask = torch.logical_and(mask, svec_mask)
        # gaussians need clone are with small spatial scale
        clone_mask = torch.logical_and(mask, torch.logical_not(split_mask))

        del mask, svec_mask

        num_split = split_mask.sum().item()
        num_clone = clone_mask.sum().item()

        console.print(f"[red bold]num_split {num_split} num_clone {num_clone}")

        # number of gaussians after split and clone will be increased by num_split + num_clone
        num_new_gaussians = num_split + num_clone

        # split
        split_mean = self.mean.data[split_mask].repeat(2, 1)
        split_qvec = self.qvec.data[split_mask].repeat(2, 1)
        split_svec = self.svec.data[split_mask].repeat(2, 1)
        # split_svec_ba = self.svec_before_activation.data[split_mask].repeat(2, 1)
        split_sh_coeffs = self.sh_coeffs.data[split_mask].repeat(2, 1, 1)
        split_alpha_ba = self.alpha_before_activation.data[split_mask].repeat(2)
        split_rotmat = qvec2rotmat_batched(split_qvec).transpose(-1, -2)

        split_gn = torch.randn(num_split * 2, 3, device=self.mean.device) * split_svec

        split_sampled_mean = split_mean + torch.einsum(
            "bij, bj -> bi", split_rotmat, split_gn
        )

        # check left product or right product

        old_num_gaussians = self.N
        unchanged_gaussians = old_num_gaussians - num_split
        self.N += num_new_gaussians
        console.print(f"[red bold]num gaussians: {self.N}")

        new_mean = torch.zeros([self.N, 3], device=self.device)
        new_qvec = torch.zeros([self.N, 4], device=self.device)
        new_svec = torch.zeros([self.N, 3], device=self.device)
        new_sh_coeffs = torch.zeros(
            [self.N, 3, self.max_C * self.max_C], device=self.device
        )
        new_alpha = torch.zeros([self.N], device=self.device)

        # copy old gaussians (# old_N - num_split)
        new_mean[:unchanged_gaussians] = self.mean.data[~split_mask]
        new_qvec[:unchanged_gaussians] = self.qvec.data[~split_mask]
        new_svec[:unchanged_gaussians] = self.svec_before_activation.data[~split_mask]
        new_sh_coeffs[:unchanged_gaussians] = self.sh_coeffs.data[~split_mask]
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
        new_sh_coeffs[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.sh_coeffs.data[clone_mask]
        new_alpha[
            unchanged_gaussians : unchanged_gaussians + num_clone
        ] = self.alpha_before_activation.data[clone_mask]

        pts = unchanged_gaussians + num_clone

        new_mean[pts : pts + 2 * num_split] = split_sampled_mean
        new_qvec[pts : pts + 2 * num_split] = split_qvec
        new_sh_coeffs[pts : pts + 2 * num_split] = split_sh_coeffs
        new_alpha[pts : pts + 2 * num_split] = split_alpha_ba
        new_svec[pts : pts + 2 * num_split] = self.svec_inv_act(
            split_svec / self.scale_shrink_factor
        )

        assert pts + 2 * num_split == self.N

        self.mean = torch.nn.Parameter(new_mean)
        self.qvec = torch.nn.Parameter(new_qvec)
        self.svec_before_activation = torch.nn.Parameter(new_svec)
        self.sh_coeffs = torch.nn.Parameter(new_sh_coeffs)
        self.alpha_before_activation = torch.nn.Parameter(new_alpha)

        del new_mean, new_qvec, new_svec, new_sh_coeffs, new_alpha
        del split_alpha_ba, split_mean, split_qvec, split_svec, split_sh_coeffs
        gc.collect()

    def remove_low_alpha_gaussians(self):
        mask = self.alpha_act(self.alpha_before_activation.data) >= self.alpha_thresh
        self.mean = torch.nn.Parameter(self.mean.data[mask])
        self.qvec = torch.nn.Parameter(self.qvec.data[mask])
        self.sh_coeffs = torch.nn.Parameter(self.sh_coeffs.data[mask])
        self.alpha_before_activation = torch.nn.Parameter(
            self.alpha_before_activation.data[mask]
        )
        self.svec_before_activation = torch.nn.Parameter(
            self.svec_before_activation.data[mask]
        )

        removed = self.N - self.mean.shape[0]
        self.N = self.mean.shape[0]
        console.print("[yellow]remove_low_alpha_gaussians[/yellow]")
        console.print(
            f"[yellow]removed {removed} gaussians, remaining {self.N} gaussians"
        )

    @torch.no_grad()
    def remove_large_gaussians(self):
        # mask for world-space gaussians
        world_space_mask = (self.svec > self.world_large_thresh).all(dim=-1)
        num_world_space_large = world_space_mask.sum().item()
        console.print(
            f"[yellow]num_world_space_large: {num_world_space_large}[/yellow]"
        )
        # TODO: remove large gaussians in view space
        # mask for view-space gaussians
        # view_space_mask = (self.radius_2d > self.view_large_thresh).all(dim=-1)
        # num_view_space_large = view_space_mask.sum().item()
        # console.print(f"[yellow]num_view_space_large: {num_view_space_large}[/yellow]")
        # # view_space_mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)

        # mask = torch.logical_or(world_space_mask, view_space_mask)

        self.select_masked_gaussians(world_space_mask)

    def reset_alpha(self):
        console.print("[yellow]reset alpha[/yellow]")
        self.alpha_before_activation.data.fill_(
            self.alpha_inv_act(self.alpha_reset_val)
        )
        with torch.no_grad():
            print_info(self.alpha, "alpha")

    def remove_tiny_gaussians(self):
        mask = (self.svec > self.svec_tiny_thresh).all(dim=-1)
        removed = self.N - mask.sum().item()
        console.print(f"[red bold]removed {removed} gaussians[/red bold]")
        self.mean = torch.nn.Parameter(self.mean.data[mask])
        self.qvec = torch.nn.Parameter(self.qvec.data[mask])
        self.sh_coeffs = torch.nn.Parameter(self.sh_coeffs.data[mask])
        self.alpha_before_activation = torch.nn.Parameter(
            self.alpha_before_activation.data[mask]
        )
        self.svec_before_activation = torch.nn.Parameter(
            self.svec_before_activation.data[mask]
        )

    def update_grads(self):
        if self.split_type == "mean_grad":
            if self.split_reduction == "max":
                self.grad_mean = torch.maximum(
                    self.grad_mean, self.mean.grad.norm(dim=-1)
                )
            elif self.split_reduction == "mean":
                self.grad_mean += self.mean.grad.norm(dim=-1)
            else:
                raise NotImplementedError
        elif self.split_type == "2d_mean_grad":
            if self.split_reduction == "max":
                self.grad_mean[self.frustum_culling_mask] = torch.maximum(
                    self.grad_mean[self.frustum_culling_mask],
                    self.mean_2d.grad.norm(dim=-1),
                )
            elif self.split_reduction == "mean":
                self.grad_mean[self.frustum_culling_mask] += self.mean_2d.grad.norm(
                    dim=-1
                )
            else:
                raise NotImplementedError

    @lineprofiler
    def adaptive_control(self, epoch, force=False):
        if epoch in self.sh_upgrades:
            self.now_C += 1
            self.now_C = min(self.now_C, self.max_C)
            console.print(
                f"Spherical Harmonics order now: {self.now_C}", style="magenta"
            )

        if epoch < self.warm_up or epoch > self.cfg.adapt_ctrl_end:
            return

        self.update_grads()

        if step_check(epoch, self.adaptive_control_iteration) or force:
            self.split_gaussians()
            self.N_changed = True

        if step_check(epoch, self.remove_low_alpha_period) or force:
            self.remove_low_alpha_gaussians()
            self.N_changed = True

        if step_check(epoch, self.alpha_reset_period) or force:
            self.reset_alpha()
            self.N_changed = True

        # if self.remove_tiny and step_check(epoch, self.remove_tiny_period):
        #     self.remove_tiny_gaussians()

        # if step_check(epoch, self.remove_large_period):
        #     self.remove_large_gaussians()
        #     self.N_changed = True

        if self.N_changed:
            self.grad_mean = torch.zeros_like(self.mean[..., 0])
            self.cnt = torch.zeros_like(self.mean[..., 0], dtype=torch.int32)
            gc.collect()
            torch.cuda.empty_cache()

            self.N_changed = False

    def save(self, path):
        path = Path(path)
        parent = path.parent
        if not parent.exists():
            parent.mkdir(parents=True)
        state_dict = {
            "mean": self.mean.data,
            "qvec": self.qvec.data,
            "svec_before_activation": self.svec_before_activation.data,
            "sh_coeffs": self.sh_coeffs.data,
            "alpha_before_activation": self.alpha_before_activation.data,
            "N": self.N,
            "cfg": self.cfg,
        }

        torch.save(state_dict, path)

    @classmethod
    def load(cls, path, cfg=None):
        state_dict = torch.load(path)

        saved_cfg = state_dict["cfg"]
        if cfg is not None:
            try:
                # saved_cfg.update(cfg)
                saved_cfg = cfg
            except:
                pass
        renderer = cls(saved_cfg)
        renderer.N = state_dict["N"]
        renderer.mean = torch.nn.Parameter(state_dict["mean"])
        renderer.qvec = torch.nn.Parameter(state_dict["qvec"])
        renderer.svec_before_activation = torch.nn.Parameter(
            state_dict["svec_before_activation"]
        )
        renderer.sh_coeffs = torch.nn.Parameter(state_dict["sh_coeffs"])
        renderer.alpha_before_activation = torch.nn.Parameter(
            state_dict["alpha_before_activation"]
        )

        renderer.grad_mean = torch.zeros_like(renderer.mean[..., 0])
        renderer.cnt = torch.zeros_like(renderer.mean[..., 0], dtype=torch.int32)

        assert renderer.N == renderer.mean.shape[0]

        return renderer

    def get_param_groups(self):
        param_groups = {
            "mean": self.mean,
            "qvec": self.qvec,
            "svec": self.svec_before_activation,
            "sh_coeffs": self.sh_coeffs,
            "alpha": self.alpha_before_activation,
        }
        return param_groups

    def get_optimizer(self, epoch):
        lr = self.cfg.lr

        opt_params = []

        param_groups = self.get_param_groups()
        for name, params in param_groups.items():
            opt_params.append({"params": params, "lr": self.scheduler[name](epoch)})

        return self.opt_cls(opt_params, lr=lr, **self.opt_args)

    def select_masked_gaussians(self, mask):
        self.mean = torch.nn.Parameter(self.mean.data[mask])
        self.qvec = torch.nn.Parameter(self.qvec.data[mask])
        self.sh_coeffs = torch.nn.Parameter(self.sh_coeffs.data[mask])
        self.alpha_before_activation = torch.nn.Parameter(
            self.alpha_before_activation.data[mask]
        )
        self.svec_before_activation = torch.nn.Parameter(
            self.svec_before_activation.data[mask]
        )
        self.N = self.mean.shape[0]

    def vis_grads_gaussians(self, thresh):
        mask = torch.zeros_like(self.mean[..., 0], dtype=torch.bool)
        mask[self.frustum_culling_mask] = self.mean_2d.grad.norm(dim=-1) > thresh
        # mask = self.mean.grad.abs().mean(dim=-1) > thresh
        self.select_masked_gaussians(mask)

    @torch.no_grad()
    def log_statistics(self, writer, epoch):
        writer.add_histogram("hists/mean", self.mean.norm(dim=-1).cpu().numpy(), epoch)
        writer.add_histogram(
            "hists/svec", self.svec.max(dim=-1)[0].cpu().numpy(), epoch
        )
        writer.add_histogram("hists/alpha", self.alpha.cpu().numpy(), epoch)
        if self.mean.grad is not None:
            writer.add_histogram(
                "hists/grad_mean", self.mean.grad.norm(dim=-1).cpu().numpy(), epoch
            )

    ## exports
    def to_pointcloud(self):
        pcd = {}
        pcd["pos"] = self.mean.data.cpu().numpy()
        pcd["rgb"] = torch.sigmoid(self.sh_coeffs.data[..., 0]).cpu().numpy()
        pcd["alpha"] = self.alpha.data.cpu().numpy()

        return pcd

    def alpha_penalty(self, type="center_weighted"):
        if type == "center_weighted":
            center = torch.zeros_like(self.mean.data[0])[None, ...]
            return (self.alpha * (self.mean.detach() - center).norm(dim=-1)).mean()
        elif type == "uniform_weighted":
            return self.alpha.norm(dim=-1).mean()
        else:
            raise NotImplementedError()

    def clip_grad(self):
        assert hasattr(self.cfg, "grad_clip"), "grad_clip not found in cfg"

        for field in self.fields:
            if getattr(self.cfg.grad_clip, field) is not None:
                # assert (
                #     getattr(self, field).grad is not None
                # ), f"{field}.grad is None when clipping"
                if field in ["svec", "alpha"]:
                    param = f"{field}_before_activation"
                else:
                    param = field
                torch.nn.utils.clip_grad_norm_(
                    getattr(self, param), self.cfg.grad_clip[field]
                )

    def force_split(self):
        """force spliting **all** gaussians, used in generative, testing"""
        split_mean = self.mean.data.repeat(2, 1)
        split_qvec = self.qvec.data.repeat(2, 1)
        split_svec = self.svec.data.repeat(2, 1)
        split_sh_coeffs = self.sh_coeffs.data.repeat(2, 1, 1)
        split_alpha_ba = self.alpha_before_activation.data.repeat(2)

        split_rotmat = qvec2rotmat_batched(split_qvec).transpose(-1, -2)

        split_gn = torch.randn(self.N * 2, 3, device=self.mean.device) * split_svec

        new_mean = split_mean + torch.einsum("bij, bj -> bi", split_rotmat, split_gn)
        new_qvec = split_qvec
        new_svec_ba = self.svec_inv_act(split_svec / self.scale_shrink_factor)

        new_sh_coeffs = split_sh_coeffs
        new_alpha_ba = split_alpha_ba

        self.mean = torch.nn.Parameter(new_mean)
        self.qvec = torch.nn.Parameter(new_qvec)
        self.svec_before_activation = torch.nn.Parameter(new_svec_ba)
        self.sh_coeffs = torch.nn.Parameter(new_sh_coeffs)
        self.alpha_before_activation = torch.nn.Parameter(new_alpha_ba)

        self.N = 2 * self.N

        self.grad_mean = torch.zeros_like(self.mean[..., 0])
        self.cnt = torch.zeros_like(self.mean[..., 0], dtype=torch.int32)
        gc.collect()
        self.N_changed = False
        torch.cuda.empty_cache()

    def remove_gaussians(self, mask, msg=None):
        # mask == 1 refers to the gaussians to be kept
        removed = self.N - mask.sum().item()
        if msg is not None:
            console.print(
                f"[red bold]removed {removed} gaussians[/red bold] according to {msg}"
            )
        self.mean = torch.nn.Parameter(self.mean.data[mask])
        self.qvec = torch.nn.Parameter(self.qvec.data[mask])
        self.sh_coeffs = torch.nn.Parameter(self.sh_coeffs.data[mask])
        self.alpha_before_activation = torch.nn.Parameter(
            self.alpha_before_activation.data[mask]
        )
        self.svec_before_activation = torch.nn.Parameter(
            self.svec_before_activation.data[mask]
        )
        self.N = self.mean.shape[0]
        self.grad_mean = torch.zeros_like(self.mean[..., 0])
        self.cnt = torch.zeros_like(self.mean[..., 0], dtype=torch.int32)
        gc.collect()
        torch.cuda.empty_cache()
