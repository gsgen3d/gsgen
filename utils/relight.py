import argparse
import numpy as np
import torch
from gs.gaussian_splatting import GaussianSplattingRenderer
from utils.ops import angle_bisector
from utils.camera import CameraInfo
from data import get_c2w_from_up_and_look_at
from utils.misc import save_img, get_ckpt_path
import imageio
from pytorch3d.structures import Pointclouds


def compute_color(light_pos, light_color, surface_normal, surface_color, mean, cam_pos):
    ab = angle_bisector(light_pos - mean, cam_pos - mean)
    # backface culling
    dot = (ab * surface_normal).sum(dim=-1).abs().clamp(min=0.0, max=1.0)

    return light_color * dot[..., None] * surface_color


@torch.no_grad()
def relight_video(ckpt, N: int = 30):
    ckpt_path = get_ckpt_path(ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    prompt = cfg["prompt"]["prompt"]

    renderer = GaussianSplattingRenderer.load(None, ckpt).to("cuda")

    renderer.eval()

    camera_info = CameraInfo.from_reso(512)
    up = np.array([0.0, 0.0, 1.0])
    look_at = np.array([0.0, 0.0, 0.0])
    pos = np.array([3.0, 0.0, 1.0])

    c2w = torch.from_numpy(get_c2w_from_up_and_look_at(up, look_at, pos)).to("cuda")

    azimuth = torch.linspace(0, 2 * np.pi, N)
    x = torch.cos(azimuth) * 3.0
    y = torch.sin(azimuth) * 3.0
    z = torch.ones_like(y) * 3.0
    light_positions = torch.stack([x, y, z], dim=-1).to("cuda")

    images = []
    light_color = torch.tensor([1.0, 1.0, 1.0], device="cuda")

    normal = (
        Pointclouds(renderer.mean.data[None, ...])
        .estimate_normals(neighborhood_size=10)[0]
        .to(renderer.mean.data)
    )

    for light_pos in light_positions:
        color = compute_color(
            light_pos,
            light_color,
            normal,
            renderer.color.data,
            renderer.mean.data,
            c2w[:3, 3],
        )
        out = renderer.render_one(
            c2w, camera_info, overrides={"color": color}, rgb_only=True
        )["rgb"]

        images.append(out)

    images = (torch.stack(images, dim=0).cpu().numpy() * 255.0).astype(np.uint8)

    imageio.mimwrite(f"./tmp/{prompt}_relight.mp4", images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="path to the checkpoint")
    parser.add_argument("--N", type=int, default=30, help="number of frames")
    args = parser.parse_args()

    relight_video(args.ckpt, args.N)
