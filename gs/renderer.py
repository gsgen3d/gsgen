from typing import Any
import cv2
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

console = Console()

try:
    import _gs as _backend
except ImportError:
    console.print("Existing gs installation not found, using loaded")
    from .backend import _backend


def step_check(step, step_size, run_at_zero=False) -> bool:
    """Returns true based on current step and step interval. credit: nerfstudio"""
    if step_size == 0:
        return False
    return (run_at_zero or step != 0) and step % step_size == 0


class Renderer:
    """
    just for test
    """

    def __init__(self, tile_size=16) -> None:
        self.tile_size = tile_size

    def culling3d(self):
        # call _C to cull
        pass

    def project_pts(self, pts: torch.Tensor, c2w: TensorType["N", 3, 4]):
        d = -c2w[..., :3, 3]
        W = torch.transpose(c2w[..., :3, :3], -1, -2)

        return torch.einsum("ij,bj->bi", W, pts + d)

    def jacobian(self, u):
        l = torch.norm(u, dim=-1)
        print(l.shape)
        print(l.max())
        print(l.min())
        J = torch.zeros(u.size(0), 3, 3).to(u)
        J[..., 0, 0] = 1.0 / u[..., 2]
        J[..., 2, 0] = u[..., 0] / l
        J[..., 1, 1] = 1.0 / u[..., 2]
        J[..., 2, 1] = u[..., 1] / l
        J[..., 0, 2] = -u[..., 0] / u[..., 2] / u[..., 2]
        J[..., 1, 2] = -u[..., 1] / u[..., 2] / u[..., 2]
        J[..., 2, 2] = u[..., 2] / l
        print_info(torch.det(J), "det(J)")

        return J

    def project_gaussian(
        self,
        mean: TensorType["N", 3],
        qvec: TensorType["N", 4],
        svec: TensorType["N", 3],
        camera: PerspectiveCameras,
        idx: int,
    ):
        projected_mean = self.project_pts(mean, camera.c2ws[idx]).contiguous()  # [N, 3]
        print_info(projected_mean[..., 2], "projected_mean_z")

        # test
        # projected_mean /= projected_mean[..., 2:]

        rotmat = qsvec2rotmat_batched(qvec, svec)
        # 3d gaussian paper eq (6)
        sigma = rotmat @ torch.transpose(rotmat, -1, -2)

        print_info(sigma, "sigma")

        W = torch.transpose(camera.c2ws[idx][:3, :3], -1, -2)
        print_info(W, "W")
        J = self.jacobian(projected_mean)
        print_info(J, "J")
        JW = torch.einsum("bij,jk->bik", J, W)

        projected_cov = torch.bmm(torch.bmm(JW, sigma), torch.transpose(JW, -1, -2))[
            ..., :2, :2
        ].contiguous()
        # projected_cov += torch.eye(2).to(projected_cov)
        print_info(projected_cov, "projected_cov")
        print_info(projected_cov[..., 0, 0], "projected_cov[..., 0, 0]")

        depth = projected_mean[..., 2:].clone().contiguous()

        projected_mean = (
            projected_mean[..., :2] / projected_mean[..., 2:]
        ).contiguous()

        return projected_mean, projected_cov, JW, depth

    def tile_partition(
        self, mean, cov_inv, camera: PerspectiveCameras, thresh: float = 0.2
    ):
        H, W = camera.h, camera.w
        print(f"H: {H} and W: {W}")
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w

        print(f"n_tiles_h: {n_tiles_h} and n_tiles_w: {n_tiles_w}")

        img_topleft = torch.FloatTensor(
            [-camera.cx / camera.fx, -camera.cy / camera.fy]
        ).to("cuda")

        img_bottomright = torch.FloatTensor(
            [(W - camera.cx) / camera.fx, (H - camera.cy) / camera.fy]
        ).to("cuda")

        print(f"img_topleft: {img_topleft}")
        print(f"img_bottomright: {img_bottomright}")

        num_gaussians = torch.zeros(n_tiles, dtype=torch.int32, device="cuda")

        pixel_size_x = 1.0 / camera.fx
        pixel_size_y = 1.0 / camera.fy

        _backend.count_num_gaussians_each_tile(
            mean,
            cov_inv,
            img_topleft,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            num_gaussians,
            thresh,
        )
        print(num_gaussians.sum().item())

        print_info(num_gaussians, "num_gaussians")

    def tile_partition_bcircle(
        self, mean, radius, camera: PerspectiveCameras, cov=None, thresh=0.1
    ):
        H, W = camera.h, camera.w
        print(f"H: {H} and W: {W}")
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        self.n_tiles_h = n_tiles_h
        self.n_tiles_w = n_tiles_w
        n_tiles = n_tiles_h * n_tiles_w

        print(f"n_tiles_h: {n_tiles_h} and n_tiles_w: {n_tiles_w}")

        img_topleft = torch.FloatTensor(
            [-camera.cx / camera.fx, -camera.cy / camera.fy]
        ).to("cuda")

        img_bottomright = torch.FloatTensor(
            [(W - camera.cx) / camera.fx, (H - camera.cy) / camera.fy]
        ).to("cuda")

        print(f"img_topleft: {img_topleft}")
        print(f"img_bottomright: {img_bottomright}")

        num_gaussians = torch.zeros(n_tiles, dtype=torch.int32, device="cuda")

        pixel_size_x = 1.0 / camera.fx
        pixel_size_y = 1.0 / camera.fy

        tic()
        _backend.count_num_gaussians_each_tile_bcircle(
            mean,
            radius,
            img_topleft,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            num_gaussians,
        )
        # _backend.count_num_gaussians_each_tile(
        #     mean,
        #     cov,
        #     img_topleft,
        #     self.tile_size,
        #     n_tiles_h,
        #     n_tiles_w,
        #     pixel_size_x,
        #     pixel_size_y,
        #     num_gaussians,
        #     thresh,
        # )
        toc()
        print(num_gaussians.sum().item())

        self.total_gaussians = num_gaussians.sum().item()
        self.num_gaussians = num_gaussians

        self.tiledepth = torch.zeros(
            self.total_gaussians, dtype=torch.float64, device="cuda"
        )

        print(f"total_gaussians: {self.total_gaussians}")
        print_info(num_gaussians, "num_gaussians")

    def image_level_radix_sort(
        self, mean, cov, radius, depth, color, camera: PerspectiveCameras
    ):
        print("=" * 10, "image_level_radix_sort", "=" * 10)
        H, W = camera.h, camera.w
        print(f"H: {H} and W: {W}")
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w

        print(f"n_tiles_h: {n_tiles_h} and n_tiles_w: {n_tiles_w}")

        img_topleft = torch.FloatTensor(
            [-camera.cx / camera.fx, -camera.cy / camera.fy]
        ).to("cuda")

        img_bottomright = torch.FloatTensor(
            [(W - camera.cx) / camera.fx, (H - camera.cy) / camera.fy]
        ).to("cuda")

        print(f"img_topleft: {img_topleft}")
        print(f"img_bottomright: {img_bottomright}")
        print(self.num_gaussians.shape)

        pixel_size_x = 1.0 / camera.fx
        pixel_size_y = 1.0 / camera.fy
        self.offset = torch.zeros(n_tiles + 1, dtype=torch.int32, device="cuda")
        print_info(self.offset, "offset")
        print_info(self.num_gaussians, "num_gaussians")
        print_info(self.tiledepth, "tiledepth")

        num_gaussians_bkp = self.num_gaussians.clone()
        gaussian_ids = torch.zeros(
            self.total_gaussians, dtype=torch.int32, device="cuda"
        )

        tic()
        _backend.prepare_image_sort(
            gaussian_ids,
            self.tiledepth,
            depth,
            self.num_gaussians,
            self.offset,
            mean,
            radius,
            img_topleft,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
        )
        # _backend.image_sort(
        #     gaussian_ids,
        #     self.tiledepth,
        #     depth,
        #     self.num_gaussians,
        #     self.offset,
        #     mean,
        #     cov,
        #     img_topleft,
        #     self.tile_size,
        #     n_tiles_h,
        #     n_tiles_w,
        #     pixel_size_x,
        #     pixel_size_y,
        #     0.01,
        # )
        toc()
        self.offset[-1] = self.total_gaussians

        print_info(self.offset, "offset")
        print_info(self.num_gaussians, "num_gaussians")
        print(f"original num_gaussians: {num_gaussians_bkp.sum().item()}")
        print(f"sorted num_gaussians: {self.num_gaussians.sum().item()}")
        n_gaussians_check = (
            (self.num_gaussians == num_gaussians_bkp).count_nonzero().item()
        )
        print(f"n_gaussians_check: {n_gaussians_check}")

        print_info(self.tiledepth, "tiledepth")
        print_info(gaussian_ids, "gaussian_ids")
        print_info(self.offset, "offset")
        diff = self.offset[1:] - self.offset[:-1]
        print_info(diff, "diff")

        out = torch.zeros([H * W * 3], dtype=torch.float32, device="cuda")
        alpha = (
            torch.ones([self.total_gaussians], dtype=torch.float32, device="cuda") * 1.0
        )

        print(self.offset[:100])
        print(gaussian_ids[:100])
        self.gaussian_ids = gaussian_ids

        print(cov.shape)
        print_info(
            cov[..., 0, 0],
            "cov[..., 0, 0]",
        )
        print_info(
            cov[..., 1, 1],
            "cov[..., 1, 1]",
        )
        print_info(torch.det(cov), "det(cov)")

        thresh = 0.001
        tic()
        _backend.tile_based_vol_rendering(
            mean,
            cov,
            # torch.inverse(cov).contiguous(),
            color,
            alpha,
            self.offset,
            gaussian_ids,
            out,
            img_topleft,
            self.tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        toc()
        # fig, ax = plt.subplots()
        print_info(out, "out")
        print(out.mean())
        print(out.std())
        # ax.imshow(out.reshape(H, W, 3).cpu().numpy())
        # plt.show()
        # fig.savefig("out.png")

        img = (out.reshape(H, W, 3).cpu().numpy() * 255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("tmp/forward_out.png", img)
        print("nans:", torch.count_nonzero(torch.isnan(out)).item())

        return out

    def render_loop(self):
        pass


@torch.no_grad()
def jacobian(u):
    l = torch.norm(u, dim=-1)
    J = torch.zeros(u.size(0), 3, 3).to(u)
    J[..., 0, 0] = 1.0 / u[..., 2]
    J[..., 2, 0] = u[..., 0] / l
    J[..., 1, 1] = 1.0 / u[..., 2]
    J[..., 2, 1] = u[..., 1] / l
    J[..., 0, 2] = -u[..., 0] / u[..., 2] / u[..., 2]
    J[..., 1, 2] = -u[..., 1] / u[..., 2] / u[..., 2]
    J[..., 2, 2] = u[..., 2] / l

    return J


@lineprofiler
def project_pts(pts: torch.Tensor, c2w: TensorType["N", 3, 4]):
    d = -c2w[..., :3, 3]
    W = torch.transpose(c2w[..., :3, :3], -1, -2)

    ret = torch.einsum("ij,bj->bi", W, pts + d)

    return ret


@lineprofiler
def project_gaussians(
    mean: TensorType["N", 3],
    qvec: TensorType["N", 4],
    svec: TensorType["N", 3],
    c2w: TensorType[3, 4],
    detach_depth: bool = False,
):
    projected_mean = project_pts(mean, c2w)
    rotmat = qsvec2rotmat_batched(qvec, svec)
    sigma = rotmat @ torch.transpose(rotmat, -1, -2)
    W = torch.transpose(c2w[:3, :3], -1, -2)
    J = jacobian(projected_mean)
    JW = torch.einsum("bij,jk->bik", J, W)
    assert JW.grad is None, "JW should not be updated"
    projected_cov = torch.bmm(torch.bmm(JW, sigma), torch.transpose(JW, -1, -2))[
        ..., :2, :2
    ].contiguous()
    # depth is always differentiable whether detach_depth is True or False
    depth = projected_mean[..., 2:].clone().contiguous()

    ###
    ### HUGE CAUTION HERE !!! the denominator here is detached
    ###
    # projected_mean = (projected_mean[..., :2] / projected_mean[..., 2:]).contiguous()
    if detach_depth:
        projected_mean = projected_mean[..., :2].contiguous() / depth.detach()
    else:
        projected_mean = projected_mean[..., :2].contiguous() / depth

    return projected_mean, projected_cov, JW, depth


class _render(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mean,
        cov,
        color,
        alpha,
        offset,
        gaussian_ids,
        topleft,
        tile_size,
        n_tiles_h,
        n_tiles_w,
        pixel_size_x,
        pixel_size_y,
        H,
        W,
        thresh,
    ):
        out = torch.zeros([H * W * 3], dtype=torch.float32, device=mean.device)
        _backend.tile_based_vol_rendering(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            out,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        ctx.save_for_backward(
            mean, cov, color, alpha, offset, gaussian_ids, out, topleft
        )
        ctx.const = [
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ]

        return out

    @staticmethod
    def backward(ctx, grad):
        mean, cov, color, alpha, offset, gaussian_ids, out, topleft = ctx.saved_tensors
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_color = torch.zeros_like(color)
        grad_alpha = torch.zeros_like(alpha)
        (
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ) = ctx.const

        _backend.tile_based_vol_rendering_backward(
            mean,
            cov,
            color,
            alpha,
            offset,
            gaussian_ids,
            out,
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            grad,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )

        return (
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _render_start_end(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mean,
        cov,
        color,
        alpha,
        start,
        end,
        gaussian_ids,
        topleft,
        tile_size,
        n_tiles_h,
        n_tiles_w,
        pixel_size_x,
        pixel_size_y,
        H,
        W,
        thresh,
    ):
        out = torch.zeros([H * W * 3], dtype=torch.float32, device=mean.device)
        _backend.tile_based_vol_rendering_start_end(
            mean,
            cov,
            color,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        ctx.save_for_backward(
            mean, cov, color, alpha, start, end, gaussian_ids, out, topleft
        )
        ctx.const = [
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ]

        return out

    @staticmethod
    def backward(ctx, grad):
        (
            mean,
            cov,
            color,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
        ) = ctx.saved_tensors
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_color = torch.zeros_like(color)
        grad_alpha = torch.zeros_like(alpha)
        (
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ) = ctx.const

        tic()
        _backend.tile_based_vol_rendering_backward_start_end(
            mean,
            cov,
            color,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            grad,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        toc("render backward")

        return (
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _render_sh(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mean,
        cov,
        sh_coeffs,
        alpha,
        start,
        end,
        gaussian_ids,
        topleft,
        c2w,
        tile_size,
        n_tiles_h,
        n_tiles_w,
        pixel_size_x,
        pixel_size_y,
        H,
        W,
        C,
        thresh,
    ):
        out = torch.zeros([H * W * 3], dtype=torch.float32, device=mean.device)
        torch.cuda.profiler.cudart().cudaProfilerStart()
        _backend.tile_based_vol_rendering_sh(
            mean,
            cov,
            sh_coeffs,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
            c2w,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            C,
            thresh,
        )
        torch.cuda.profiler.cudart().cudaProfilerStop()
        ctx.save_for_backward(
            mean,
            cov,
            sh_coeffs,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
            c2w,
        )
        ctx.const = [
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            C,
            thresh,
        ]

        return out

    @staticmethod
    def backward(ctx, grad):
        (
            mean,
            cov,
            sh_coeffs,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
            c2w,
        ) = ctx.saved_tensors

        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_sh_coeffs = torch.zeros_like(sh_coeffs)
        grad_alpha = torch.zeros_like(alpha)
        (
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            C,
            thresh,
        ) = ctx.const

        tic()
        torch.cuda.profiler.cudart().cudaProfilerStart()
        _backend.tile_based_vol_rendering_backward_sh(
            mean,
            cov,
            sh_coeffs,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            grad_mean,
            grad_cov,
            grad_sh_coeffs,
            grad_alpha,
            grad,
            topleft,
            c2w,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            C,
            thresh,
        )
        torch.cuda.profiler.cudart().cudaProfilerStop()
        toc("render backward")

        # print_info(grad_mean, "grad_mean")

        return (
            grad_mean,
            grad_cov,
            grad_sh_coeffs,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _render_sh_bg(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mean,
        cov,
        sh_coeffs,
        alpha,
        start,
        end,
        gaussian_ids,
        topleft,
        c2w,
        tile_size,
        n_tiles_h,
        n_tiles_w,
        pixel_size_x,
        pixel_size_y,
        H,
        W,
        C,
        thresh,
        bg_rgb,
    ):
        out = torch.zeros([H * W * 3], dtype=torch.float32, device=mean.device)
        torch.cuda.profiler.cudart().cudaProfilerStart()
        _backend.tile_based_vol_rendering_sh_with_bg(
            mean,
            cov,
            sh_coeffs,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
            c2w,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            C,
            thresh,
            bg_rgb,
        )
        torch.cuda.profiler.cudart().cudaProfilerStop()
        ctx.save_for_backward(
            mean,
            cov,
            sh_coeffs,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
            c2w,
            bg_rgb,
        )
        ctx.const = [
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            C,
            thresh,
        ]

        return out

    @staticmethod
    def backward(ctx, grad):
        (
            mean,
            cov,
            sh_coeffs,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
            c2w,
            bg_rgb,
        ) = ctx.saved_tensors

        grad = grad.contiguous()
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_sh_coeffs = torch.zeros_like(sh_coeffs)
        grad_alpha = torch.zeros_like(alpha)
        (
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            C,
            thresh,
        ) = ctx.const

        tic()
        torch.cuda.profiler.cudart().cudaProfilerStart()
        _backend.tile_based_vol_rendering_backward_sh_with_bg(
            mean,
            cov,
            sh_coeffs,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            grad_mean,
            grad_cov,
            grad_sh_coeffs,
            grad_alpha,
            grad,
            topleft,
            c2w,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            C,
            thresh,
            bg_rgb,
        )
        torch.cuda.profiler.cudart().cudaProfilerStop()
        toc("render backward")

        # print_info(grad_mean, "grad_mean")

        return (
            grad_mean,
            grad_cov,
            grad_sh_coeffs,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _render_scalar(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mean,
        cov,
        scalar,
        alpha,
        start,
        end,
        gaussian_ids,
        topleft,
        tile_size,
        n_tiles_h,
        n_tiles_w,
        pixel_size_x,
        pixel_size_y,
        H,
        W,
        thresh,
        T,
    ):
        out = torch.zeros([H * W], dtype=torch.float32, device=mean.device)
        _backend.tile_based_vol_rendering_scalar(
            mean,
            cov,
            scalar,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
            T,
        )
        ctx.save_for_backward(
            mean, cov, scalar, alpha, start, end, gaussian_ids, out, topleft
        )
        ctx.const = [
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ]

        return out

    @staticmethod
    def backward(ctx, grad):
        (
            mean,
            cov,
            color,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
        ) = ctx.saved_tensors
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_color = torch.zeros_like(color)
        grad_alpha = torch.zeros_like(alpha)
        (
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ) = ctx.const

        tic()
        _backend.tile_based_vol_rendering_scalar_backward(
            mean,
            cov,
            color,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            grad,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        toc("render backward")

        return (
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _render_with_T(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mean,
        cov,
        scalar,
        alpha,
        start,
        end,
        gaussian_ids,
        topleft,
        tile_size,
        n_tiles_h,
        n_tiles_w,
        pixel_size_x,
        pixel_size_y,
        H,
        W,
        thresh,
        bg,
    ):
        out = torch.zeros([H, W, 3], dtype=torch.float32, device=mean.device)
        T = torch.ones_like(out[..., :1])
        _backend.tile_based_vol_rendering_start_end_with_T(
            mean,
            cov,
            scalar,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
            T,
        )

        if torch.isnan(out).any():
            console.print("[red]In line 1181")
        out = out + T * bg

        if torch.isnan(bg).any():
            breakpoint()
            console.print("[red]bg!!!!")
        if torch.isnan(T).any():
            console.print("[red]T!!!!")

        ctx.save_for_backward(
            mean, cov, scalar, alpha, start, end, gaussian_ids, out, topleft, T
        )
        ctx.const = [
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ]

        if torch.isnan(out).any():
            console.print("[red]In line 1197")
        return out

    @staticmethod
    def backward(ctx, grad):
        (
            mean,
            cov,
            color,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            topleft,
            T,
        ) = ctx.saved_tensors
        grad = grad.contiguous()
        grad_mean = torch.zeros_like(mean)
        grad_cov = torch.zeros_like(cov)
        grad_color = torch.zeros_like(color)
        grad_alpha = torch.zeros_like(alpha)
        (
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        ) = ctx.const

        tic()
        _backend.tile_based_vol_rendering_backward_start_end(
            mean,
            cov,
            color,
            alpha,
            start,
            end,
            gaussian_ids,
            out,
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            grad,
            topleft,
            tile_size,
            n_tiles_h,
            n_tiles_w,
            pixel_size_x,
            pixel_size_y,
            H,
            W,
            thresh,
        )
        toc("render backward")

        return (
            grad_mean,
            grad_cov,
            grad_color,
            grad_alpha,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            torch.nan_to_num(grad * T),
        )


render = _render.apply
render_start_end = _render_start_end.apply
render_sh = _render_sh.apply
render_sh_bg = _render_sh_bg.apply
render_scalar = _render_scalar.apply
render_with_T = _render_with_T.apply


class GaussianRenderer(torch.nn.Module):
    def __init__(self, cfg, pts, rgb):
        super().__init__()
        self.device = cfg.device
        self.cfg = cfg
        self.N = pts.shape[0]
        self.mean = torch.nn.parameter.Parameter(pts)
        self.qvec = torch.nn.Parameter(torch.zeros([self.N, 4]))
        self.qvec.data[..., 0] = 1.0

        self.svec_before_activation = torch.nn.Parameter(torch.ones([self.N, 3]))
        self.color_before_activation = torch.nn.Parameter(rgb)
        self.alpha_before_activation = torch.nn.Parameter(torch.ones([self.N]))

        self.svec_act = activations[cfg.svec_act]
        self.color_act = activations[cfg.color_act]
        self.alpha_act = activations[cfg.alpha_act]

        self.svec_inv_act = inv_activations[cfg.svec_act]
        self.color_inv_act = inv_activations[cfg.color_act]
        self.alpha_inv_act = inv_activations[cfg.alpha_act]

        self.svec_before_activation.data.fill_(self.svec_inv_act(cfg.svec_init))
        self.color_before_activation.data = self.color_inv_act(rgb)
        self.alpha_before_activation.data.fill_(self.alpha_inv_act(cfg.alpha_init))

        self.set_cfg(cfg)

    def set_cfg(self, cfg):
        # camera imaging params
        # !! deprecated: should be provided in `camera_info` when calling forward
        # self.near_plane = cfg.near_plane
        # self.far_plane = cfg.far_plane
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
        self.adaptive_control_iteration = cfg.adaptive_control_iteration
        self.pos_grad_thresh = cfg.pos_grad_thresh
        self.split_scale_thresh = cfg.split_scale_thresh
        self.scale_shrink_factor = cfg.scale_shrink_factor
        self.alpha_reset_period = cfg.alpha_reset_period
        self.alpha_reset_val = cfg.alpha_reset_val
        self.alpha_thresh = cfg.alpha_thresh

    def render_lecacy(self, c2w, camera_info):
        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
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
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.svec[mask].contiguous()
        color = self.color[mask].contiguous()
        alpha = self.alpha[mask].contiguous()

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        # depth = depth.squeeze()
        # depth = depth.max() - depth + 0.3

        cov = (cov + cov.transpose(-1, -2)) / 2.0
        with torch.no_grad():
            m = (cov[..., 0, 0] + cov[..., 1, 1]) / 2.0
            p = torch.det(cov)
            radius = torch.sqrt(m + torch.sqrt((m.pow(2) - p).clamp(min=0.0)))

        self.depth = depth
        self.radius = radius

        if self.cfg.debug:
            print_info(radius, "radius")

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w

        if self.cfg.debug:
            print("n_tiles", n_tiles)

        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)

        num_gaussians = torch.zeros(n_tiles, dtype=torch.int32, device=self.device)
        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        tic()
        with torch.no_grad():
            if self.tile_culling_type == "bcircle":
                _backend.count_num_gaussians_each_tile_bcircle(
                    mean,
                    radius * self.tile_culling_radius,
                    img_topleft,
                    self.tile_size,
                    n_tiles_h,
                    n_tiles_w,
                    pixel_size_x,
                    pixel_size_y,
                    num_gaussians,
                )
            elif self.tile_culling_type == "prob":
                _backend.count_num_gaussians_each_tile(
                    mean,
                    cov,
                    img_topleft,
                    self.tile_size,
                    n_tiles_h,
                    n_tiles_w,
                    pixel_size_x,
                    pixel_size_y,
                    num_gaussians,
                    self.tile_culling_thresh,
                )
            else:
                raise NotImplementedError
        toc("tile culling")

        self.total_dub_gaussians = num_gaussians.sum().item()

        tiledepth = torch.zeros(
            self.total_dub_gaussians, dtype=torch.float64, device=self.device
        )
        offset = torch.zeros(n_tiles + 1, dtype=torch.int32, device=self.device)
        gaussian_ids = torch.zeros(
            self.total_dub_gaussians, dtype=torch.int32, device=self.device
        )

        tic()
        with torch.no_grad():
            if self.tile_culling_type == "bcircle":
                _backend.prepare_image_sort(
                    gaussian_ids,
                    tiledepth,
                    depth,
                    num_gaussians,
                    offset,
                    mean,
                    radius * self.tile_culling_radius,
                    img_topleft,
                    self.tile_size,
                    n_tiles_h,
                    n_tiles_w,
                    pixel_size_x,
                    pixel_size_y,
                )
            elif self.tile_culling_type == "prob":
                _backend.image_sort(
                    gaussian_ids,
                    tiledepth,
                    depth,
                    num_gaussians,
                    offset,
                    mean,
                    cov,
                    img_topleft,
                    self.tile_size,
                    n_tiles_h,
                    n_tiles_w,
                    pixel_size_x,
                    pixel_size_y,
                    self.tile_culling_thresh,
                )
            else:
                raise NotImplementedError
        toc("radix sort")
        offset[-1] = self.total_dub_gaussians

        # if self.cfg.debug:
        #     _backend.debug_check_tiledepth(offset.cpu(), tiledepth.cpu())

        tic()
        out = render(
            mean,
            cov,
            color,
            alpha,
            offset,
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
        ).view(H, W, 3)
        toc("render")

        return out

    @lineprofiler
    def render_aabb_culling(self, c2w, camera_info):
        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
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
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.svec[mask].contiguous()
        color = self.color[mask].contiguous()
        alpha = self.alpha[mask].contiguous()

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

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

        tic()
        out = render_start_end(
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
        ).view(H, W, 3)
        toc("render")

        return out

    def forward(self, c2w, camera_info):
        if self.cfg.tile_culling_type == "aabb":
            return self.render_aabb_culling(c2w, camera_info)
        else:
            return self.render_lecacy(c2w, camera_info)

    @property
    def svec(self):
        return self.svec_act(self.svec_before_activation)

    @property
    def color(self):
        return self.color_act(self.color_before_activation)

    @property
    def alpha(self):
        return self.alpha_act(self.alpha_before_activation)

    def split_gaussians(self):
        assert self.mean.grad is not None, "mean.grad is None"
        console.print("[red bold]Splitting Gaussians")
        mean_mask = self.mean.grad.norm(dim=-1) > self.pos_grad_thresh
        svec_mask = (self.svec.data > self.split_scale_thresh).any(dim=-1)
        split_mask = torch.logical_and(mean_mask, svec_mask)
        clone_mask = torch.logical_and(mean_mask, torch.logical_not(split_mask))

        num_split = split_mask.sum().item()
        num_clone = clone_mask.sum().item()

        console.print(f"[red bold]num_split {num_split} num_clone {num_clone}")

        num_new_gaussians = num_split + num_clone

        split_mean = self.mean.data[split_mask].repeat(2, 1)
        split_qvec = self.qvec.data[split_mask].repeat(2, 1)
        split_svec = self.svec.data[split_mask].repeat(2, 1)
        # split_svec_ba = self.svec_before_activation.data[split_mask].repeat(2, 1)
        split_color_ba = self.color_before_activation.data[split_mask].repeat(2, 1)
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
        new_color = torch.zeros([self.N, 3], device=self.device)
        new_alpha = torch.zeros([self.N], device=self.device)

        # copy old gaussians
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
        new_color[pts : pts + 2 * num_split] = split_color_ba
        new_alpha[pts : pts + 2 * num_split] = split_alpha_ba
        new_svec[pts : pts + 2 * num_split] = self.svec_inv_act(
            split_svec / self.scale_shrink_factor
        )

        self.mean = torch.nn.Parameter(new_mean)
        self.qvec = torch.nn.Parameter(new_qvec)
        self.svec_before_activation = torch.nn.Parameter(new_svec)
        self.color_before_activation = torch.nn.Parameter(new_color)
        self.alpha_before_activation = torch.nn.Parameter(new_alpha)

    def remove_low_alpha_gaussians(self):
        mask = self.alpha_act(self.alpha_before_activation.data) >= self.alpha_thresh
        self.mean = torch.nn.Parameter(self.mean.data[mask])
        self.qvec = torch.nn.Parameter(self.qvec.data[mask])
        self.color_before_activation = torch.nn.Parameter(
            self.color_before_activation.data[mask]
        )
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

    def reset_alpha(self):
        console.print("[yellow]reset alpha[/yellow]")
        print(self.alpha_inv_act)
        self.alpha_before_activation.data.fill_(
            self.alpha_inv_act(self.alpha_reset_val)
        )
        with torch.no_grad():
            print_info(self.alpha, "alpha")

    def adaptive_control(self, iteration):
        if step_check(iteration + 1, self.alpha_reset_period):
            self.remove_low_alpha_gaussians()
        if step_check(iteration, self.alpha_reset_period, run_at_zero=False):
            self.reset_alpha()

        if step_check(iteration, self.adaptive_control_iteration):
            self.split_gaussians()

    def check_grad(self):
        print_info(self.qvec.grad, "grad_qvec")
        print_info(self.svec_before_activation.grad, "grad_svec")

    def check_info(self):
        # print_info(self.qvec, "qvec")
        # print_info(self.svec, "svec")
        pass

    def log_n_gaussian_dub(self, writer, step):
        writer.add_scalar("n_gaussian_dub", self.total_dub_gaussians, step)

    @torch.no_grad()
    def log_grad_bounds(self, writer, step):
        if self.mean.grad is None:
            return
        writer.add_scalar("grad_bounds/mean_max", self.mean.grad.max(), step)
        writer.add_scalar("grad_bounds/mean_min", self.mean.grad.min(), step)
        writer.add_scalar("grad_bounds/qvec_max", self.qvec.grad.max(), step)
        writer.add_scalar("grad_bounds/qvec_min", self.qvec.grad.min(), step)
        writer.add_scalar(
            "grad_bounds/svec_max", self.svec_before_activation.grad.max(), step
        )
        writer.add_scalar(
            "grad_bounds/svec_min", self.svec_before_activation.grad.min(), step
        )
        writer.add_scalar(
            "grad_bounds/color_max", self.color_before_activation.grad.max(), step
        )
        writer.add_scalar(
            "grad_bounds/color_min", self.color_before_activation.grad.min(), step
        )
        writer.add_scalar(
            "grad_bounds/alpha_max", self.alpha_before_activation.grad.max(), step
        )
        writer.add_scalar(
            "grad_bounds/alpha_min", self.alpha_before_activation.grad.min(), step
        )

    @torch.no_grad()
    def log_info(self, writer, step):
        writer.add_scalar("info/mean_mean", self.mean.abs().mean(), step)
        writer.add_scalar("info/qvec_mean", self.qvec.abs().mean(), step)
        writer.add_scalar("info/svec_mean", self.svec.abs().mean(), step)
        writer.add_scalar("info/color_mean", self.color.abs().mean(), step)
        writer.add_scalar("info/alpha_mean", self.alpha.sigmoid().mean(), step)

    @torch.no_grad()
    def log_bounds(self, writer, step):
        """log the bounds of the parameters"""
        writer.add_scalar("bounds/mean_max", self.mean.max(), step)
        writer.add_scalar("bounds/mean_min", self.mean.min(), step)
        writer.add_scalar("bounds/qvec_max", self.qvec.max(), step)
        writer.add_scalar("bounds/qvec_min", self.qvec.min(), step)
        writer.add_scalar("bounds/svec_max", self.svec.max(), step)
        writer.add_scalar("bounds/svec_min", self.svec.min(), step)
        writer.add_scalar("bounds/color_max", self.color.max(), step)
        writer.add_scalar("bounds/color_min", self.color.min(), step)
        writer.add_scalar("bounds/alpha_max", self.alpha.max(), step)
        writer.add_scalar("bounds/alpha_min", self.alpha.min(), step)

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
    def log(self, writer, step):
        self.log_bounds(writer, step)
        self.log_info(writer, step)
        self.log_grad_bounds(writer, step)
        self.log_n_gaussian_dub(writer, step)
        self.log_depth_and_radius(writer, step)

    def sanity_check(self):
        print()

    def save(self, path):
        state_dict = {
            "mean": self.mean.data,
            "qvec": self.qvec.data,
            "log_svec": self.log_svec.data,
            "color": self.color.data,
            "alpha": self.alpha.data,
            "N": self.N,
            "cfg": self.cfg,
        }

        torch.save(state_dict, path)

    @classmethod
    def load(self, path):
        state_dict = torch.load(path)
        renderer = Renderer(
            state_dict["cfg"],
        )
