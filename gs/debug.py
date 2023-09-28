from typing import Any
import cv2
from utils.camera import PerspectiveCameras, CameraInfo
from utils.transforms import qsvec2rotmat_batched
from utils import misc
from utils.misc import print_info, tic, toc, save_img
from utils.vis.basic import draw_heatmap_of_num_gaussians_per_tile
import numpy as np
from torchtyping import TensorType
import torch
import matplotlib.pyplot as plt
from culling import tile_culling_aabb_count
from utils.transforms import qvec2rotmat_batched

try:
    import _gs as _backend
except ImportError:
    from .backend import _backend

from .renderer import (
    render,
    project_gaussians,
    render_start_end,
    render_sh,
    render_sh_bg,
    render_scalar,
)
from .sh_renderer import SHRenderer


def get_c2w_from_up_and_look_at(up, look_at, pos):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    y = np.cross(z, x)

    c2w = np.zeros([3, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos

    return c2w


class MockRenderer(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        misc._timing_ = True
        self.mean = torch.nn.Parameter(
            torch.FloatTensor([[0.0, 0.0, 0.0], [0.1, 0.07, 0.0]])
        )
        self.qvec = torch.nn.Parameter(
            torch.FloatTensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        )
        self.log_svec = torch.nn.Parameter(
            torch.FloatTensor([[1.0, 1.0, 0.5], [1.1, 0.5, 1.1]])
        )
        self.log_svec.data *= np.log(20e-3)
        self.color = torch.nn.Parameter(
            torch.FloatTensor([[0.01, 0.01, 0.99], [0.01, 0.99, 0.01]])
        )
        self.alpha = torch.nn.Parameter(torch.FloatTensor([10000, 10000]))

        self.D = 3.0

        layout = cfg.get("layout", "default")

        if layout == "one":
            self.mean = torch.nn.Parameter(self.mean[:1])
            self.qvec = torch.nn.Parameter(self.qvec[:1])
            self.color = torch.nn.Parameter(self.color[:1])
            self.alpha = torch.nn.Parameter(self.alpha[:1])
            self.log_svec = torch.nn.Parameter(self.log_svec[:1])

        if layout == "one_repeat":
            self.mean = torch.nn.Parameter(self.mean[:1].repeat(100, 1))
            self.qvec = torch.nn.Parameter(self.qvec[:1].repeat(100, 1))
            self.color = torch.nn.Parameter(self.color[:1].repeat(100, 1))
            self.alpha = torch.nn.Parameter(self.alpha[:1].repeat(100, 1))
            self.log_svec = torch.nn.Parameter(self.log_svec[:1].repeat(100, 1))

        if layout == "one_repeat":
            self.mean = torch.nn.Parameter(self.mean.repeat(100, 1))
            self.qvec = torch.nn.Parameter(self.qvec.repeat(100, 1))
            self.color = torch.nn.Parameter(self.color.repeat(100, 1))
            self.alpha = torch.nn.Parameter(self.alpha.repeat(100, 1))
            self.log_svec = torch.nn.Parameter(self.log_svec.repeat(100, 1))

        if layout == "grid":
            x = torch.linspace(-5, -10, 100)
            y = torch.linspace(-10, 10, 40)
            z = torch.linspace(-10, 10, 50)
            x, y, z = torch.meshgrid(x, y, z)
            x = x.reshape(-1)
            y = y.reshape(-1)
            z = z.reshape(-1)
            mean = torch.stack([x, y, z], dim=1)
            self.mean = torch.nn.Parameter(mean)
            self.N = self.mean.shape[0]
            self.qvec = torch.nn.Parameter(
                torch.FloatTensor([[1.0, 0.0, 0.0, 0.0]] * self.N)
            )
            self.log_svec = torch.nn.Parameter(
                torch.FloatTensor([[1.0, 1.0, 1.0]] * self.N)
            )
            self.log_svec.data *= np.log(120e-3)
            color = torch.randn(self.N, 3).clamp(min=0.0, max=1.0)
            # color = torch.linspace(0, 1, 3 * self.N, dtype=torch.float32).reshape(
            #     self.N, 3
            # )
            self.color = torch.nn.Parameter(color)
            self.alpha = torch.nn.Parameter(torch.FloatTensor([0.0] * self.N))

        self.cfg = cfg
        self.device = cfg.device
        self.N = self.mean.shape[0]
        self.near_plane = cfg.near_plane
        self.far_plane = cfg.far_plane
        self.tile_size = cfg.tile_size
        self.frustum_culling_radius = cfg.frustum_culling_radius
        self.tile_culling_type = cfg.tile_culling_type
        self.tile_culling_radius = cfg.tile_culling_radius
        self.tile_culling_thresh = cfg.tile_culling_thresh
        self.T_thresh = cfg.T_thresh

    def forward(self, c2w, camera_info):
        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.log_svec.exp(),
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.log_svec[mask].exp().contiguous()
        color = self.color[mask].contiguous()
        # alpha = self.alpha[mask].contiguous()
        # alpha = torch.nn.functional.sigmoid(self.alpha[mask].contiguous())
        alpha = torch.sigmoid(self.alpha[mask].contiguous())

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        cov = (cov + cov.transpose(-1, -2)) / 2.0
        with torch.no_grad():
            m = (cov[..., 0, 0] + cov[..., 1, 1]) / 2.0
            p = torch.det(cov)
            radius = torch.sqrt(m + torch.sqrt((m.pow(2) - p).clamp(min=0.0)))

        if self.cfg.debug:
            print_info(radius, "radius")

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
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

        self.num_gaussians_bkp = num_gaussians.clone()

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
        print("total dub gaussians", self.total_dub_gaussians)
        offset[-1] = self.total_dub_gaussians
        print_info(num_gaussians, "num_gaussians")
        self.offset = offset
        self.num_gaussians = num_gaussians
        self.gaussian_ids = gaussian_ids

        self.mean_2d = mean.detach().clone()
        self.radius_2d = radius.detach().clone() * self.tile_culling_radius
        self.cov_2d = cov.detach().clone()

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

    def render_with_aabb_culling(self, c2w, camera_info):
        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.log_svec.exp(),
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.log_svec[mask].exp().contiguous()
        color = self.color[mask].contiguous()
        # alpha = self.alpha[mask].contiguous()
        # alpha = torch.nn.functional.sigmoid(self.alpha[mask].contiguous())
        alpha = torch.sigmoid(self.alpha[mask].contiguous())

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        tic()
        N_with_dub, aabb_topleft, aabb_bottomright = tile_culling_aabb_count(
            mean, cov, self.tile_size, camera_info, 9
        )
        print("N_with_dub", N_with_dub)
        toc("count N with dub")
        print(aabb_bottomright.dtype)
        print(aabb_topleft.dtype)

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        print("n_tiles", n_tiles)
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)
        offset = torch.zeros([n_tiles + 1], dtype=torch.int32, device=self.device)
        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy
        gaussian_ids = torch.zeros([N_with_dub], dtype=torch.int32, device=self.device)

        tic()
        _backend.tile_culling_aabb(
            aabb_topleft,
            aabb_bottomright,
            gaussian_ids,
            offset,
            depth,
            n_tiles_h,
            n_tiles_w,
        )
        toc("tile culling aabb")

        for i in range(n_tiles, -1, -1):
            if offset[i].item() == -1:
                offset[i] = offset[i + 1]

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

    def test_basic_alias(self):
        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            1297,
            840,
            0.0,
            1000,
        )

        camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)

        out = self.forward(c2w, camera_info)
        torch.nn.functional.mse_loss(out, torch.zeros_like(out)).backward()

        print_info(out, "out")

        img = (out.cpu().detach().numpy() * 255.0).astype(np.uint8)

        # color = (255, 255, 255)
        # mean_2d = self.mean_2d.cpu().numpy()
        # radius_2d = (self.radius_2d.cpu().numpy() * camera_info.fx).astype(np.int32)

        # for i in range(self.N):
        #     x, y = (int(mean_2d[i, 0] * camera_info.fx + camera_info.cx), int(mean_2d[i, 1] * camera_info.fy + camera_info.cy))
        #     img = cv2.circle(img, (x, y), radius_2d[i], color, 1)

        # # print(img.max())
        # # print(img.min())

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)

        # n_tiles = n_tiles_h * n_tiles_w

        # num_error_samples = 0

        # for t in range(n_tiles):
        #     start = self.offset[t]
        #     end = self.offset[t + 1]
        #     assert end - start <= 2
        #     if end - start == 2:
        #         if self.gaussian_ids[start] == self.gaussian_ids[start + 1]:
        #             num_error_samples += 1

        # print("num_error_samples", num_error_samples)

        heatmap = draw_heatmap_of_num_gaussians_per_tile(
            "test_alias_heatmap.png",
            self.tile_size,
            self.num_gaussians.cpu().numpy(),
            n_tiles_h,
            n_tiles_w,
            H,
            W,
            False,
        )
        # for i in range(self.N):
        #     x, y = (int(mean_2d[i, 0] * camera_info.fx + camera_info.cx), int(mean_2d[i, 1] * camera_info.fy + camera_info.cy))
        #     heatmap = cv2.circle(heatmap, (x, y), radius_2d[i], color, 1)

        cv2.imwrite("./tmp/test_basic_alias.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite("./tmp/test_basic_alias_heatmap.png", heatmap)

        # print(self.mean_2d[0])
        # print(self.cov_2d[0])

        # print(self.mean_2d[1])
        # print(self.cov_2d[1])
        print_info(self.num_gaussians, "num_gaussians")

        assert (self.num_gaussians == self.num_gaussians_bkp).all()

    def test_aabb(self):
        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            1297,
            840,
            0.0,
            1000,
        )

        camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)

        out = self.forward(c2w, camera_info)

        img = (out.cpu().detach().numpy() * 255.0).astype(np.uint8)

        color = (255, 255, 255)
        mean_2d = self.mean_2d.cpu().numpy()
        cov_2d = self.cov_2d.cpu().numpy()
        center = camera_info.camera_space_to_pixel_space(mean_2d)
        aabb_x = np.sqrt(cov_2d[..., 0, 0] * 5) * camera_info.fx
        aabb_y = np.sqrt(cov_2d[..., 1, 1] * 5) * camera_info.fy

        for i in range(self.N):
            top_left = (int(center[i, 0] - aabb_x[i]), int(center[i, 1] - aabb_y[i]))
            bottom_right = (
                int(center[i, 0] + aabb_x[i]),
                int(center[i, 1] + aabb_y[i]),
            )
            img = cv2.rectangle(img, top_left, bottom_right, color, 3)

        cv2.imwrite("./tmp/test_aabb.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def test_render_with_aabb_culling(self):
        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            1297,
            840,
            0.0,
            1000,
        )

        camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)

        out = self.render_with_aabb_culling(c2w, camera_info)
        print_info(out, "out")
        img = (out.cpu().detach().numpy() * 255.0).astype(np.uint8)
        cv2.imwrite(
            "./tmp/test_render_with_aabb_culling.png",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

    def benchmark_render(self):
        from .benchmarks import _render_v0, _render_v1, _render_v2

        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            1297,
            840,
            0.0,
            1000,
        )

        camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)
        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.log_svec.exp(),
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.log_svec[mask].exp().contiguous()
        color = self.color[mask].contiguous()
        # alpha = self.alpha[mask].contiguous()
        # alpha = torch.nn.functional.sigmoid(self.alpha[mask].contiguous())
        alpha = torch.sigmoid(self.alpha[mask].contiguous())

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        cov = (cov + cov.transpose(-1, -2)) / 2.0
        with torch.no_grad():
            m = (cov[..., 0, 0] + cov[..., 1, 1]) / 2.0
            p = torch.det(cov)
            radius = torch.sqrt(m + torch.sqrt((m.pow(2) - p).clamp(min=0.0)))

        if self.cfg.debug:
            print_info(radius, "radius")

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
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

        self.num_gaussians_bkp = num_gaussians.clone()

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
        print("total dub gaussians", self.total_dub_gaussians)
        offset[-1] = self.total_dub_gaussians
        print_info(num_gaussians, "num_gaussians")
        self.offset = offset
        self.num_gaussians = num_gaussians
        self.gaussian_ids = gaussian_ids

        self.mean_2d = mean.detach().clone()
        self.radius_2d = radius.detach().clone() * self.tile_culling_radius
        self.cov_2d = cov.detach().clone()

        # if self.cfg.debug:
        #     _backend.debug_check_tiledepth(offset.cpu(), tiledepth.cpu())

        tic()
        for _ in range(100):
            out = _render_v0.apply(
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
        toc("render v0")

        tic()
        for _ in range(100):
            out = _render_v1.apply(
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
        toc("render v1")

        tic()
        for _ in range(100):
            out = _render_v2.apply(
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
        toc("render v2")

        return out

    def benchmark_culling(self):
        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            1297,
            840,
            0.0,
            1000,
        )

        camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)

        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.log_svec.exp(),
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.log_svec[mask].exp().contiguous()
        color = self.color[mask].contiguous()
        # alpha = self.alpha[mask].contiguous()
        # alpha = torch.nn.functional.sigmoid(self.alpha[mask].contiguous())
        alpha = torch.sigmoid(self.alpha[mask].contiguous())

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)

        with torch.no_grad():
            m = (cov[..., 0, 0] + cov[..., 1, 1]) / 2.0
            p = torch.det(cov)
            radius = torch.sqrt(m + torch.sqrt((m.pow(2) - p).clamp(min=0.0)))

        num_gaussians = torch.zeros(n_tiles, dtype=torch.int32, device=self.device)
        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

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
        self.total_dub_gaussians = num_gaussians.sum().item()
        self.num_gaussians_bkp = num_gaussians.clone()
        tiledepth = torch.zeros(
            self.total_dub_gaussians, dtype=torch.float64, device=self.device
        )
        offset = torch.zeros(n_tiles + 1, dtype=torch.int32, device=self.device)
        gaussian_ids = torch.zeros(
            self.total_dub_gaussians, dtype=torch.int32, device=self.device
        )

        tic()
        for _ in range(10):
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
        toc("bcircle culling")

        N_with_dub, aabb_topleft, aabb_bottomright = tile_culling_aabb_count(
            mean, cov, self.tile_size, camera_info, 9
        )

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)
        offset = torch.zeros([n_tiles + 1], dtype=torch.int32, device=self.device)
        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy
        gaussian_ids = torch.zeros([N_with_dub], dtype=torch.int32, device=self.device)

        tic()
        for _ in range(10):
            _backend.tile_culling_aabb(
                aabb_topleft,
                aabb_bottomright,
                gaussian_ids,
                offset,
                depth,
                n_tiles_h,
                n_tiles_w,
            )
        toc("aabb culling")

    def complete_render_aabb_culling(self):
        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            1297,
            840,
            0.0,
            1000,
        )

        # camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)
        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.log_svec.exp(),
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.log_svec[mask].exp().contiguous()
        color = self.color[mask].contiguous()
        alpha = torch.sigmoid(self.alpha[mask].contiguous())

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        tic()
        N_with_dub, aabb_topleft, aabb_bottomright = tile_culling_aabb_count(
            mean, cov, self.tile_size, camera_info, 10
        )
        print("N_with_dub", N_with_dub)
        toc("count N with dub")
        print(aabb_bottomright.dtype)
        print(aabb_topleft.dtype)

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        print("n_tiles", n_tiles)
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)
        # offset = torch.zeros([n_tiles + 1], dtype=torch.int32, device=self.device)
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

        start_bkp = start.clone()
        end_bkp = end.clone()

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

        print((start_bkp == start).all())
        print((end_bkp == end).all())
        # breakpoint()

        print_info(cov, "cov")
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

        print(out.shape)
        print_info(out, "out")
        img = (out.cpu().detach().numpy() * 255.0).astype(np.uint8)
        cv2.imwrite(
            "./tmp/complete_render_aabb_culling.png",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

    def sh_render_sanity_check(self):
        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            1297,
            840,
            0.0,
            1000,
        )

        sh_order = self.cfg.get("sh_order", 1)

        sh_base = 0.28209479177387814
        self.color = torch.nn.Parameter(
            torch.logit(self.color.data.unsqueeze(-1).repeat(1, 1, sh_order * sh_order))
            / sh_base
        )
        # self.color = torch.nn.Parameter(torch.ones_like(self.color) / sh_base)
        print(self.color.shape)
        self.to(self.device)

        # camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)

        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.log_svec.exp(),
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.log_svec[mask].exp().contiguous()
        color = self.color[mask].contiguous()
        alpha = torch.sigmoid(self.alpha[mask].contiguous())

        # n_alive = color.shape[0]

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        tic()
        N_with_dub, aabb_topleft, aabb_bottomright = tile_culling_aabb_count(
            mean, cov, self.tile_size, camera_info, 10
        )
        print("N_with_dub", N_with_dub)
        toc("count N with dub")

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        print("n_tiles", n_tiles)
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)
        # offset = torch.zeros([n_tiles + 1], dtype=torch.int32, device=self.device)
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

        print_info(cov, "cov")
        out = render_sh(
            mean,
            cov,
            color,
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
            sh_order,
            self.T_thresh,
        ).view(H, W, 3)

        print_info(out, "out")
        img = (out.cpu().detach().numpy() * 255.0).astype(np.uint8)

        color = (255, 255, 255)
        mean_2d = mean.detach().cpu().numpy()
        cov_2d = cov.detach().cpu().numpy()
        center = camera_info.camera_space_to_pixel_space(mean_2d)
        aabb_x = np.sqrt(cov_2d[..., 0, 0] * 10) * camera_info.fx
        aabb_y = np.sqrt(cov_2d[..., 1, 1] * 10) * camera_info.fy

        for i in range(self.N):
            top_left = (int(center[i, 0] - aabb_x[i]), int(center[i, 1] - aabb_y[i]))
            bottom_right = (
                int(center[i, 0] + aabb_x[i]),
                int(center[i, 1] + aabb_y[i]),
            )
            img = cv2.rectangle(img, top_left, bottom_right, color, 3)

        cv2.imwrite(
            "./tmp/sh_render_sanity_check.png",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )

        num_gaussian_per_tile = (end - start).cpu().numpy()
        heatmap = draw_heatmap_of_num_gaussians_per_tile(
            "sh_render_heatmap.png",
            self.tile_size,
            num_gaussian_per_tile,
            n_tiles_h,
            n_tiles_w,
            H,
            W,
            False,
        )

        ## backward sanity test

        target = torch.zeros_like(out)

        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()

    def test_split(self):
        num_split = 1
        split_mask = torch.ones([1], dtype=torch.bool, device=self.device)

        NN = 16

        split_mean = self.mean.data[split_mask].squeeze().repeat(NN, 1)
        split_qvec = self.qvec.data[split_mask].squeeze().repeat(NN, 1)
        split_log_svec = self.log_svec.data[split_mask].squeeze().repeat(NN, 1)

        split_rotmat = qvec2rotmat_batched(split_qvec).transpose(-1, -2)

        split_color = self.color.data[split_mask].squeeze().repeat(NN, 1)
        split_alpha = self.alpha.data[split_mask].squeeze().repeat(NN)

        split_gn = (
            torch.randn(num_split * NN, 3, device=self.mean.device)
            * split_log_svec.exp()
        )

        split_sampled_mean = split_mean + torch.einsum(
            "bij, bj -> bi", split_rotmat, split_gn
        )

        self.mean = torch.nn.Parameter(split_sampled_mean)
        self.qvec = torch.nn.Parameter(split_qvec)
        self.log_svec = torch.nn.Parameter(torch.log(split_log_svec.exp() / NN))
        self.color = torch.nn.Parameter(split_color)
        self.alpha = torch.nn.Parameter(split_alpha)

        self.N = NN

        self.sh_render_sanity_check()

    def sh_render_bg(self):
        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            1297,
            840,
            0.0,
            1000,
        )

        sh_order = self.cfg.get("sh_order", 1)

        sh_base = 0.28209479177387814
        self.color = torch.nn.Parameter(
            torch.logit(self.color.data.unsqueeze(-1).repeat(1, 1, sh_order * sh_order))
            / sh_base
        )
        # self.color = torch.nn.Parameter(torch.ones_like(self.color) / sh_base)
        print(self.color.shape)
        self.to(self.device)

        # camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)

        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.log_svec.exp(),
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.log_svec[mask].exp().contiguous()
        color = self.color[mask].contiguous()
        alpha = torch.sigmoid(self.alpha[mask].contiguous())

        # n_alive = color.shape[0]

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        tic()
        N_with_dub, aabb_topleft, aabb_bottomright = tile_culling_aabb_count(
            mean, cov, self.tile_size, camera_info, 10
        )
        print("N_with_dub", N_with_dub)
        toc("count N with dub")

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        print("n_tiles", n_tiles)
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)
        # offset = torch.zeros([n_tiles + 1], dtype=torch.int32, device=self.device)
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

        self.bg_rgb = [1.0, 1.0, 1.0]
        self.bg_rgb = torch.FloatTensor(self.bg_rgb).to(self.device)

        print_info(cov, "cov")
        out = render_sh_bg(
            mean,
            cov,
            color,
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
            sh_order,
            self.T_thresh,
            self.bg_rgb,
        ).view(H, W, 3)

        print_info(out, "out")
        img = (out.cpu().detach().numpy() * 255.0).astype(np.uint8)

        cv2.imwrite("./tmp/sh_render_bg.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def render_depth(self):
        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        H = 840
        W = 1297
        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            W,
            H,
            0.0,
            1000,
        )

        sh_base = 0.28209479177387814
        # self.color = torch.nn.Parameter(torch.ones_like(self.color) / sh_base)
        print(self.color.shape)
        self.to(self.device)

        # camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)

        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.log_svec.exp(),
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.log_svec[mask].exp().contiguous()
        color = self.color[mask].contiguous()
        alpha = torch.sigmoid(self.alpha[mask].contiguous())

        # n_alive = color.shape[0]

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        tic()
        N_with_dub, aabb_topleft, aabb_bottomright = tile_culling_aabb_count(
            mean, cov, self.tile_size, camera_info, 10
        )
        print("N_with_dub", N_with_dub)
        toc("count N with dub")

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        print("n_tiles", n_tiles)
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)
        # offset = torch.zeros([n_tiles + 1], dtype=torch.int32, device=self.device)
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

        self.bg_rgb = [1.0, 1.0, 1.0]
        self.bg_rgb = torch.FloatTensor(self.bg_rgb).to(self.device)

        print_info(cov, "cov")

        T = torch.zeros([H, W, 1], dtype=torch.float32, device=self.device)
        out = render_scalar(
            mean,
            cov,
            depth.detach(),
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
        ).view(H, W, 1)

        img = (out.cpu().detach().numpy() * 255.0).astype(np.uint8)
        cv2.imwrite("./tmp/rendered_depth.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


class DebugRenderer(SHRenderer):
    def __init__(self, cfg, pts=None, rgb=None):
        layout = cfg.get("layout", "default")

        mean = torch.nn.Parameter(
            torch.FloatTensor([[0.0, 0.0, 0.0], [0.1, 0.07, 0.0]])
        )
        qvec = torch.nn.Parameter(
            torch.FloatTensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        )
        log_svec = torch.nn.Parameter(
            torch.FloatTensor([[1.0, 1.0, 0.5], [1.1, 0.5, 1.1]])
        )
        log_svec.data *= np.log(20e-3)
        color = torch.nn.Parameter(
            torch.FloatTensor([[0.01, 0.01, 0.99], [0.01, 0.99, 0.01]])
        )
        alpha = torch.nn.Parameter(torch.FloatTensor([10000, 10000]))

        if layout == "one":
            mean = torch.nn.Parameter(mean[:1])
            qvec = torch.nn.Parameter(qvec[:1])
            color = torch.nn.Parameter(color[:1])
            alpha = torch.nn.Parameter(alpha[:1])
            log_svec = torch.nn.Parameter(log_svec[:1])

        super().__init__(cfg, mean, color)

    def test_bg(self):
        pass

    def render_depth(self):
        up = np.array([0, 0, 1], dtype=np.float32)
        look_at = np.array([0, 0, 0], dtype=np.float32)
        pos = np.array([1, 0, 0], dtype=np.float32)

        H = 840
        W = 1297
        camera_info = CameraInfo(
            961.22,
            963.09,
            648.38,
            420.12,
            W,
            H,
            0.0,
            1000,
        )

        sh_base = 0.28209479177387814
        # self.color = torch.nn.Parameter(torch.ones_like(self.color) / sh_base)
        print(self.color.shape)
        self.to(self.device)

        # camera_info.upsample(4)

        c2w = get_c2w_from_up_and_look_at(up, look_at, pos)
        c2w = torch.from_numpy(c2w).to(self.device)

        f_normals, f_pts = camera_info.get_frustum(c2w)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _backend.culling_gaussian_bsphere(
                self.mean,
                self.qvec,
                self.log_svec.exp(),
                f_normals,
                f_pts,
                mask,
                self.frustum_culling_radius,
            )
        mean = self.mean[mask].contiguous()
        qvec = self.qvec[mask].contiguous()
        svec = self.log_svec[mask].exp().contiguous()
        color = self.color[mask].contiguous()
        alpha = torch.sigmoid(self.alpha[mask].contiguous())

        # n_alive = color.shape[0]

        pixel_size_x = 1.0 / camera_info.fx
        pixel_size_y = 1.0 / camera_info.fy

        mean, cov, JW, depth = project_gaussians(mean, qvec, svec, c2w)

        tic()
        N_with_dub, aabb_topleft, aabb_bottomright = tile_culling_aabb_count(
            mean, cov, self.tile_size, camera_info, 10
        )
        print("N_with_dub", N_with_dub)
        toc("count N with dub")

        H, W = camera_info.h, camera_info.w
        n_tiles_h = H // self.tile_size + (H % self.tile_size > 0)
        n_tiles_w = W // self.tile_size + (W % self.tile_size > 0)
        n_tiles = n_tiles_h * n_tiles_w
        print("n_tiles", n_tiles)
        img_topleft = torch.FloatTensor(
            [-camera_info.cx / camera_info.fx, -camera_info.cy / camera_info.fy],
        ).to(self.device)
        # offset = torch.zeros([n_tiles + 1], dtype=torch.int32, device=self.device)
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

        self.bg_rgb = [1.0, 1.0, 1.0]
        self.bg_rgb = torch.FloatTensor(self.bg_rgb).to(self.device)

        print_info(cov, "cov")

        T = torch.zeros([H, W, 1], dtype=torch.float32, device=self.device)
        out = render_scalar(
            mean,
            cov,
            depth.detach(),
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
        ).view(H, W, 1)

        img = (out.cpu().detach().numpy() * 255.0).astype(np.uint8)
        cv2.imwrite("./tmp/rendered_depth.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
