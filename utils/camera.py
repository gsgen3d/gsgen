import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .colmap import (
    read_images,
    read_cameras,
    read_pts_from_colmap,
    read_images_test,
    read_images_v1,
    read_one_image,
)
from .misc import print_info, tic, toc
from .transforms import rotmat2wxyz
from rich.console import Console

console = Console()

"""Using OpenCV coordinates"""


class PerspectiveCameras:
    def __init__(self, c2ws, fx, fy, cx, cy, w, h, distortion=None) -> None:
        self.c2ws = c2ws
        self.n_cams = self.c2ws.shape[0]
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h

        self.yfov = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect = w / h
        self.distortion = distortion

    def to(self, device):
        self.c2ws = self.c2ws.to(device)
        self.c2ws = self.c2ws.to(torch.float32)

    def get_frustum_pts(self, idx, near_plane, far_plane):
        if idx < 0 or idx >= self.n_cams:
            raise ValueError

        c2w = self.c2ws[idx]

        up = -c2w[:, 1]
        right = c2w[:, 0]
        lookat = c2w[:, 2]
        t = c2w[:, 3]

        half_vside = far_plane * np.tan(self.yfov * 0.5)
        half_hside = half_vside * self.aspect

        near_point = near_plane * lookat + t
        far_point = far_plane * lookat + t
        near_normal = lookat
        far_normal = -lookat

        left_normal = torch.cross(far_point - half_hside * right, up)
        right_normal = torch.cross(far_point + half_hside * right, up)

        up_normal = torch.cross(right, far_point + half_vside * up)
        down_normal = torch.cross(right, far_point - half_vside * up)

        corners = []
        for a in [-1, 1]:
            for b in [-1, 1]:
                corners.append(
                    near_point
                    + a * half_hside * near_plane / far_plane * right
                    + b * half_vside * near_plane / far_plane * up
                )

        for a in [-1, 1]:
            for b in [-1, 1]:
                corners.append(far_point + a * half_hside * right + b * half_vside * up)

        return corners

    # depreatied
    # def get_tile_frustum(self, idx, near_plane, far_plane, tile_size=16):
    #     if not tile_size == 16:
    #         raise NotImplementedError

    def get_frustum(self, idx, near_plane, far_plane):
        if idx < 0 or idx >= self.n_cams:
            raise ValueError

        c2w = self.c2ws[idx]

        up = -c2w[:, 1]
        right = c2w[:, 0]
        lookat = c2w[:, 2]
        t = c2w[:, 3]

        half_vside = far_plane * np.tan(self.yfov * 0.5)
        half_hside = half_vside * self.aspect

        near_point = near_plane * lookat
        far_point = far_plane * lookat
        near_normal = lookat
        far_normal = -lookat

        left_normal = torch.cross(far_point - half_hside * right, up)
        right_normal = torch.cross(up, far_point + half_hside * right)

        up_normal = torch.cross(far_point + half_vside * up, right)
        down_normal = torch.cross(right, far_point - half_vside * up)

        pts = [near_point + t, far_point + t, t, t, t, t]
        normals = [
            near_normal,
            far_normal,
            left_normal,
            right_normal,
            up_normal,
            down_normal,
        ]

        pts = torch.stack(pts, dim=0)
        normals = torch.stack(normals, dim=0)
        normals = F.normalize(normals, dim=-1)

        return normals, pts

    def get_all_rays_o(self, idx):
        xp = (torch.arange(0, self.w, dtype=torch.float32) - self.cx) / self.fx
        yp = (torch.arange(0, self.h, dtype=torch.float32) - self.cy) / self.fy

        xp, yp = torch.meshgrid(xp, yp, indexing="ij")
        xp = xp.reshape(-1)
        yp = yp.reshape(-1)
        padding = torch.ones_like(xp)

        xyz_cam = torch.stack([xp, yp, padding], dim=-1)

        rot = self.c2ws[idx][:3, :3]
        t = self.c2ws[idx][:3][3]

        xyz_world = t + torch.einsum("ij,bj->bi", rot, xyz_cam)

        return xyz_world

    def prepare(self):
        pixel_size_x = 1.0 / self.fx
        pixel_size_y = 1.0 / self.fy

    def get_camera_wxyz(self, idx: int):
        return rotmat2wxyz(self.c2ws[idx][:3, :3].contiguous())

    def get_camera_pos(self, idx: int):
        return self.c2ws[idx][:3, 3]


def get_data(cfg):
    base = Path(cfg.data_dir)
    cam_bin = base / "colmap" / "sparse" / "0" / "cameras.bin"
    image_bin = base / "colmap" / "sparse" / "0" / "images.bin"
    point_bin = base / "colmap" / "sparse" / "0" / "points3D.bin"
    rot, t, images = read_images(image_bin, cfg.image_dir)
    pts, rgb = read_pts_from_colmap(
        point_bin,
    )
    t = np.expand_dims(t, axis=-1)
    camera = read_cameras(cam_bin)
    params = camera.params
    # print(rot.shape)
    # print(t.shape)
    # print(images.shape)

    rot = rot.transpose(0, 2, 1)
    t = -rot @ t
    c2ws = np.concatenate([rot, t], axis=2)
    c2ws = torch.from_numpy(c2ws)

    cams = None

    if camera.model == "OPENCV":
        cams = PerspectiveCameras(
            c2ws,
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            camera.width,
            camera.height,
        )
    elif camera.model == "PINHOLE":
        cams = PerspectiveCameras(
            c2ws,
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            camera.width,
            camera.height,
        )
    elif camera.model == "OPENCV_FISHEYE":
        # TODO: add fisheye camera support
        cams = PerspectiveCameras(
            c2ws,
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            camera.width,
            camera.height,
        )
    else:
        print("Not support camera model: ", camera.model)
        raise NotImplementedError

    print(
        f"camera:\n\tfx: {cams.fx}; fy: {cams.fy}\n\tcx: {cams.cx}; cy: {cams.cy}\n\tH: {cams.h}; W: {cams.w}"
    )

    return cams, images, pts, rgb


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

    def set_reso(self, reso: int):
        self.fx = reso
        self.fy = reso
        self.cx = reso / 2.0
        self.cy = reso / 2.0
        self.w = reso
        self.h = reso

        self.yfov = 2 * np.arctan(self.h / (2 * self.fy))
        self.aspect = self.w / self.h

    def get_frustum(self, c2w):
        up = -c2w[:, 1]
        right = c2w[:, 0]
        lookat = c2w[:, 2]
        t = c2w[:, 3]

        half_vside = self.far_plane * np.tan(self.yfov * 0.5)
        half_hside = half_vside * self.aspect

        near_point = self.near_plane * lookat
        far_point = self.far_plane * lookat
        near_normal = lookat
        far_normal = -lookat

        left_normal = torch.cross(far_point - half_hside * right, up)
        right_normal = torch.cross(up, far_point + half_hside * right)

        up_normal = torch.cross(far_point + half_vside * up, right)
        down_normal = torch.cross(right, far_point - half_vside * up)

        pts = [near_point + t, far_point + t, t, t, t, t]
        normals = [
            near_normal,
            far_normal,
            left_normal,
            right_normal,
            up_normal,
            down_normal,
        ]

        pts = torch.stack(pts, dim=0)
        normals = torch.stack(normals, dim=0)
        normals = F.normalize(normals, dim=-1)

        return normals, pts

    def print_camera_info(self):
        console.print(
            f"[blue underline]camera:\n\tfx: {self.fx:.2f}; fy: {self.fy:.2f}\n\tcx: {self.cx:.2f}; cy: {self.cy:.2f}\n\tH: {self.h}; W: {self.w}\n\tpixel_size: {1 / self.fx:.4g} pixel_size_y: {1 / self.fy:.4g}"
        )

    def camera_space_to_pixel_space(self, pts):
        if pts.shape[1] == 3:
            pts = pts[:, :2] / pts[:, 2:]

        assert pts.shape[1] == 2

        pts[:, 0] = pts[:, 0] * self.fx + self.cx
        pts[:, 1] = pts[:, 1] * self.fy + self.cy
        if isinstance(pts, np.ndarray):
            pts = pts.astype(np.int32)
        elif isinstance(pts, torch.Tensor):
            pts = pts.to(torch.int32)

        return pts

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

        xyz_cam = torch.stack([xp, yp, padding], dim=-1).to(c2w.device)

        rot = c2w[:3, :3]

        return (
            torch.einsum("ij,bj->bi", rot, xyz_cam)
            .reshape(
                self.w, self.h, 3
            )  # NOTE: first (w, h), and then transpose for consistency with `indexing="ij"`
            .transpose(0, 1)
        ).to(c2w.device)

    @classmethod
    def from_reso(cls, reso: int):
        return cls(
            reso,
            reso,
            reso / 2.0,
            reso / 2.0,
            reso,
            reso,
            0.01,
            1000,
        )

    def get_camera_intrinsic(self, device="cuda"):
        return torch.tensor(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1],
            ]
        ).to(device)


def in_frustum(queries, normal, pts):
    is_in = torch.ones_like(queries[..., 0], dtype=torch.bool)
    for i in range(6):
        in_test = torch.einsum("bj,j->b", queries - pts[i], normal[i]) > 0.0
        is_in = torch.logical_and(is_in, in_test)

    return is_in


def get_c2ws_and_camera_info(cfg):
    console.print("[bold green]Loading camera info...")
    base = Path(cfg.data_dir)
    cam_bin = base / "colmap" / "sparse" / "0" / "cameras.bin"
    image_bin = base / "colmap" / "sparse" / "0" / "images.bin"
    point_bin = base / "colmap" / "sparse" / "0" / "points3D.bin"
    points_cached = base / "colmap" / "sparse" / "0" / "points_and_rgb.pt"
    console.print("[bold green]Loading images...")
    rot, t, images = read_images(image_bin, cfg.image_dir)
    console.print("[bold green]Loading points...")
    if points_cached.exists():
        console.print("[bold green]Loading cached points...")
        pts, rgb = torch.load(points_cached)
    else:
        pts, rgb = read_pts_from_colmap(
            point_bin,
        )
        pts = pts.astype(np.float32)
        rgb = rgb.astype(np.float32)
        pts = torch.from_numpy(pts)
        rgb = torch.from_numpy(rgb)
        torch.save((pts, rgb), points_cached)
    t = np.expand_dims(t, axis=-1)
    camera = read_cameras(cam_bin)
    params = camera.params
    # print(rot.shape)
    # print(t.shape)
    # print(images.shape)

    rot = rot.transpose(0, 2, 1)
    t = -rot @ t
    c2ws = np.concatenate([rot, t], axis=2).astype(np.float32)
    c2ws = torch.from_numpy(c2ws)

    cams = None
    console.print(f"[green bold]camera model: {camera.model}")

    if camera.model == "OPENCV" or camera.model == "PINHOLE":
        camera_info = CameraInfo(
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            camera.width,
            camera.height,
            cfg.near_plane,
            cfg.far_plane,
        )
    elif camera.model == "OPENCV_FISHEYE":
        # TODO: add fisheye camera support
        raise NotImplementedError
    else:
        print("Not support camera model: ", camera.model)
        raise NotImplementedError

    console.print(f"[blue underline]downsample: {cfg.downsample}")

    camera_info.downsample(cfg.downsample)

    assert images.shape[0] == c2ws.shape[0]
    if (
        abs(camera_info.w - images.shape[2]) > 1
        or abs(camera_info.h - images.shape[1]) > 1
    ):
        console.print("[red bold]camera image size not match")
        exit(-1)

    if (camera_info.h != images.shape[1]) or (camera_info.w != images.shape[2]):
        console.print(
            "[red bold]camera image size not match due to round caused by downsample"
        )
        camera_info.h = images.shape[1]
        camera_info.w = images.shape[2]

    console.print(
        f"[blue underline]camera:\n\tfx: {camera_info.fx:.2f}; fy: {camera_info.fy:.2f}\n\tcx: {camera_info.cx:.2f}; cy: {camera_info.cy:.2f}\n\tH: {camera_info.h}; W: {camera_info.w}"
    )

    if isinstance(pts, np.ndarray):
        pts = pts.astype(np.float32)
        rgb = rgb.astype(np.float32)
        pts = torch.from_numpy(pts)
        rgb = torch.from_numpy(rgb)
    c2ws = c2ws.to(cfg.device)

    return c2ws, camera_info, images, pts, rgb


def get_eval_mask(data_dir, filenames):
    data_dir = Path(data_dir)
    test_meta = data_dir / "transforms_test.json"
    with open(test_meta, "r") as f:
        test_meta = json.load(f)

    eval_filenames = []

    for ff in test_meta["frames"]:
        fpath = Path(ff["file_path"])
        eval_filenames.append(fpath.name)

    eval_mask = np.isin(filenames, eval_filenames)

    return torch.from_numpy(eval_mask)


def get_c2ws_and_camera_info_v1(cfg):
    console.print("[bold green]Loading camera info...")
    base = Path(cfg.data_dir)
    cam_bin = base / "colmap" / "sparse" / "0" / "cameras.bin"
    image_bin = base / "colmap" / "sparse" / "0" / "images.bin"
    point_bin = base / "colmap" / "sparse" / "0" / "points3D.bin"
    points_cached = base / "colmap" / "sparse" / "0" / "points_and_rgb.pt"
    console.print("[bold green]Loading images...")

    image_dirname = "images"

    if cfg.downsample > 1:
        image_dirname += f"_{cfg.downsample}"

    tic()
    rot, t, images, filenames = read_images_v1(image_bin, base / image_dirname)
    toc("read images v1")

    if cfg.eval_type == "mipnerf":
        eval_mask = torch.zeros(images.shape[0], dtype=torch.bool)
        eval_mask[::8] = True
    elif cfg.eval_type == "nerfstudio":
        eval_mask = get_eval_mask(base, filenames)
    else:
        raise NotImplementedError

    console.print("[bold green]Loading points...")
    if points_cached.exists():
        console.print("[bold green]Loading cached points...")
        pts, rgb = torch.load(points_cached)
    else:
        pts, rgb = read_pts_from_colmap(
            point_bin,
        )
        pts = pts.astype(np.float32)
        rgb = rgb.astype(np.float32)
        pts = torch.from_numpy(pts)
        rgb = torch.from_numpy(rgb)
        torch.save((pts, rgb), points_cached)
    t = np.expand_dims(t, axis=-1)
    camera = read_cameras(cam_bin)
    params = camera.params

    rot = rot.transpose(0, 2, 1)
    t = -rot @ t
    c2ws = np.concatenate([rot, t], axis=2).astype(np.float32)
    c2ws = torch.from_numpy(c2ws)

    cams = None
    console.print(f"[green bold]camera model: {camera.model}")

    if camera.model == "OPENCV" or camera.model == "PINHOLE":
        camera_info = CameraInfo(
            float(params[0]),
            float(params[1]),
            float(params[2]),
            float(params[3]),
            camera.width,
            camera.height,
            cfg.near_plane,
            cfg.far_plane,
        )
    elif camera.model == "OPENCV_FISHEYE":
        # TODO: add fisheye camera support
        raise NotImplementedError
    else:
        print("Not support camera model: ", camera.model)
        raise NotImplementedError

    console.print(f"[blue underline]downsample: {cfg.downsample}")

    camera_info.downsample(cfg.downsample)

    assert images.shape[0] == c2ws.shape[0]
    if (
        abs(camera_info.w - images.shape[2]) > 1
        or abs(camera_info.h - images.shape[1]) > 1
    ):
        console.print("[red bold]camera image size not match")
        exit(-1)

    if (camera_info.h != images.shape[1]) or (camera_info.w != images.shape[2]):
        console.print(
            "[red bold]camera image size not match due to round caused by downsample"
        )
        camera_info.h = images.shape[1]
        camera_info.w = images.shape[2]

    console.print(
        f"[blue underline]camera:\n\tfx: {camera_info.fx:.2f}; fy: {camera_info.fy:.2f}\n\tcx: {camera_info.cx:.2f}; cy: {camera_info.cy:.2f}\n\tH: {camera_info.h}; W: {camera_info.w}"
    )

    if isinstance(pts, np.ndarray):
        pts = pts.astype(np.float32)
        rgb = rgb.astype(np.float32)
        pts = torch.from_numpy(pts)
        rgb = torch.from_numpy(rgb)
    c2ws = c2ws.to(cfg.device)

    return c2ws, camera_info, images, pts, rgb, eval_mask


def read_from_json(cfg):
    downsample = cfg.downsample
    base_dir = Path(cfg.data_dir)
    assert base_dir.exists()

    c2ws = []
    images = []

    H = 800
    W = 800

    with open(base_dir / "transforms_train.json", "r") as f:
        meta = json.load(f)

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    camera_info = CameraInfo(
        focal,
        focal,
        0.5 * W,
        0.5 * H,
        W,
        H,
        cfg.near_plane,
        cfg.far_plane,
    )

    N_train = len(meta["frames"])
    for ff in meta["frames"]:
        c2w = np.array(ff["transform_matrix"])
        img_name = str(Path(ff["file_path"]).name)
        c2ws.append(c2w)
        images.append(read_one_image(base_dir / "train" / f"{img_name}.png"))

    with open(base_dir / "transforms_test.json", "r") as f:
        meta = json.load(f)
    N_eval = len(meta["frames"])
    for ff in meta["frames"]:
        c2w = np.array(ff["transform_matrix"])
        img_name = str(Path(ff["file_path"]).name)
        c2ws.append(c2w)
        images.append(read_one_image(base_dir / "test" / f"{img_name}.png"))

    eval_mask = np.zeros(N_train + N_eval, dtype=np.bool)
    eval_mask[N_train:] = True
    c2ws = np.array(c2ws)
    images = np.array(images)

    if downsample > 1:
        images = images[:, ::downsample, ::downsample, :]
        camera_info.downsample(downsample)

    return c2ws, images, camera_info, eval_mask


def get_c2ws_and_camera_info_nerf_sythetic(cfg):
    console.print("Using synthetic data...", style="magenta")
    console.print("[bold green]Loading camera info and randomly init points and rgb...")

    c2ws, images, camera_info, eval_mask = read_from_json(cfg)
    c2ws = torch.from_numpy(c2ws).to(torch.float32)[:, :3]
    # OpenGL c2ws to OpenCV c2ws (flip y and z axis)
    c2ws[..., :3, 1] = -c2ws[..., :3, 1]
    c2ws[..., :3, 2] = -c2ws[..., :3, 2]
    images = torch.from_numpy(images).to(torch.float32)
    eval_mask = torch.from_numpy(eval_mask).to(torch.bool)

    num_points = int(cfg.num_points)

    xyz = torch.randn((num_points, 3)).to(torch.float32)
    xyz = xyz.clamp(-cfg.bounds, cfg.bounds)

    rgb = torch.rand((num_points, 3)).to(torch.float32)
    rgb = rgb.clamp(0.001, 0.999)

    return c2ws, camera_info, images, xyz, rgb, eval_mask
