import numpy as np
from pathlib import Path
import torch
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor,
    AlphaCompositor,
    FoVOrthographicCameras,
    look_at_view_transform,
)
from .colmap import read_images, read_cameras, read_pts_from_colmap
from .misc import save_fig, print_info
import warnings
import visdom


def get_data(cfg):
    """
    get data from colmap and convert to pytorch3d format, assuming only one camera type
    """
    base = Path(cfg.data_dir)
    cam_bin = base / "colmap" / "sparse" / "0" / "cameras.bin"
    image_bin = base / "colmap" / "sparse" / "0" / "images.bin"
    point_bin = base / "colmap" / "sparse" / "0" / "points3D.bin"
    rot, t, images = read_images(image_bin, cfg.image_dir)
    pts, rgb = read_pts_from_colmap(
        point_bin,
    )
    camera = read_cameras(cam_bin)
    params = camera.params
    rot = torch.from_numpy(rot.astype(np.float32)).to("cuda")
    t = torch.from_numpy(t.astype(np.float32)).to("cuda")
    print(t[0])

    rot = rot.transpose(-1, -2)
    t = -torch.einsum("bij,bj->bi", rot, t)

    if camera.model == "OPENCV":
        focal_length = [[float(params[0]), float(params[1])]]
        principal_point = ((float(params[2]), float(params[3])),)
        image_size = ((camera.height, camera.width),)
        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=image_size,
            R=rot,
            T=t,
            device="cuda",
            in_ndc=False,
        )
    elif camera.model == "OPENCV_FISHEYE":
        warnings.warn(
            "Fisheye camera model currently not supported. This function will return a perspective camera."
        )
        focal_length = [[float(params[0]), float(params[1])]]
        principal_point = ((float(params[2]), float(params[3])),)
        image_size = ((camera.height, camera.width),)
        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=image_size,
            R=rot,
            T=t,
            device="cuda",
            in_ndc=False,
        )
    else:
        raise NotImplementedError

    print(cameras)

    return cameras, images, pts, rgb


def render_pcd(
    cameras: PerspectiveCameras,
    xyz: np.ndarray,
    rgb: np.ndarray,
    idx: int,
    radius: float = 0.003,
    points_per_pixel: int = 10,
):
    vis = visdom.Visdom(env="render_pcd")
    cam = cameras[idx]
    image_size = int(cam.get_image_size()[0][0].item()), int(
        cam.get_image_size()[0][1].item()
    )
    print(image_size)

    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel,
    )

    xyz = torch.from_numpy(xyz).to(cam.device).to(torch.float32)
    rgb = torch.from_numpy(rgb).to(cam.device).to(torch.float32)
    pcd = Pointclouds(points=[xyz], features=[rgb])

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cam, raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=(0, 0, 0)),
    )

    image = renderer(pcd).cpu().squeeze().moveaxis(-1, 0).numpy()[:, ::-1, ::-1]

    print(image.max())
    print(image.min())

    vis.image(image, win="render_pcd")

    save_fig(image, f"render_pcd_{idx}")
