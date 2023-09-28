import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
import viser
import time
import viser
import viser.transforms as tf
import cv2
from collections import deque


class CameraInfo:
    def __init__(self, fx, fy, cx, cy, w, h, near_plane, far_plane) -> None:
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h

        self.yfov = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect = w / h
        self.near_plane = near_plane
        self.far_plane = far_plane

    def downsample(self, scale):
        self.fx /= scale
        self.fy /= scale
        self.cx /= scale
        self.cy /= scale
        self.w //= scale
        self.h //= scale

        self.yfov = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect = self.w / self.h

    def upsample(self, scale):
        self.fx *= scale
        self.fy *= scale
        self.cx *= scale
        self.cy *= scale
        self.w *= int(scale)
        self.h *= int(scale)

    @classmethod
    def from_fov_camera(cls, fov, aspect, resolution, near_plane, far_plane):
        W = resolution
        H = int(resolution / aspect)
        cx = W / 2
        cy = H / 2
        fx = cx / np.tan(fov / 2)
        fy = cy / np.tan(fov / 2)

        return cls(fx, fy, cx, cy, W, H, near_plane, far_plane)

    def get_rays_d(self, c2w):
        xp = (torch.arange(0, self.w, dtype=torch.float32) - self.cx) / self.fx
        yp = (torch.arange(0, self.h, dtype=torch.float32) - self.cy) / self.fy

        xp, yp = torch.meshgrid(xp, yp, indexing="ij")
        xp = xp.reshape(-1)
        yp = yp.reshape(-1)
        padding = torch.ones_like(xp)

        xyz_cam = torch.stack([xp, yp, padding], dim=-1)

        rot = c2w[:3, :3]

        return (
            torch.einsum("ij,bj->bi", rot, xyz_cam)
            .reshape(
                self.w, self.h, 3
            )  # NOTE: first (w, h), and then transpose for consistency with `indexing="ij"`
            .transpose(0, 1)
        )


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def get_c2w(camera):
    c2w = np.zeros([3, 4], dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position

    return c2w


default_cfg = {
    "port": 4321,
    "device": "cpu",
}
default_cfg = OmegaConf.create(default_cfg)


class PointCloudViewer:
    def __init__(self, cfg=None) -> None:
        if cfg is None:
            cfg = default_cfg
        self.port = cfg.port
        self.device = cfg.device
        self.server = viser.ViserServer(port=cfg.port)

        self.render_times = deque(maxlen=3)
        self.need_update = False

        self.pcd_size = self.server.add_gui_slider(
            "Point Cloud Radius", 0.05, 0.1, 0.01, 0.05
        )
        self.reset_view_button = self.server.add_gui_button("Reset View")

        @self.pcd_size.on_update
        def _(_):
            self.need_update = True

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

    def set_pointcloud(self, pcd):
        if isinstance(pcd, torch.Tensor):
            pcd = pcd.detach().cpu().numpy()

        self.xyz = pcd[..., :3]
        self.rgb = pcd[..., 3:]

    def update(self):
        assert hasattr(
            self, "xyz"
        ), "Please set point cloud first (using `set_pointcloud` method)"
        if self.need_update:
            self.server.add_point_cloud(
                "/pointcloud",
                self.xyz,
                self.rgb,
                self.pcd_size.value,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument(
        "--N", type=int, default=-1, help="max points to visualize, -1 for all"
    )
    parser.add_argument("--port", type=int, default=4321)
    parser.add_argument("--device", type=str, default="cuda")

    opt = parser.parse_args()

    viewer = PointCloudViewer(opt)

    params = torch.load(opt.ckpt, map_location=opt.device)["params"]
    xyz = params["mean"]
    rgb = torch.sigmoid(params["color"])

    if opt.N > 0:
        from utils.ops import farthest_point_sampling

        _, indices = farthest_point_sampling(xyz, opt.N)
        xyz = xyz[indices]
        rgb = rgb[indices]

    viewer.set_pointcloud(torch.cat([xyz, rgb], dim=-1))
    while True:
        viewer.update()
        time.sleep(0.01)
