import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from gs.gaussian_splatting import GaussianSplattingRenderer
from utils.spiral import get_c2w_from_up_and_look_at, get_camera_path_fixed_elevation
from utils.camera import CameraInfo
from tqdm import tqdm
from utils.colormaps import apply_depth_colormap
from mediapy import write_video
from imageio import mimwrite
from einops import repeat

from rich.console import Console

console = Console()


def take_spiral_from_ckpt(
    ckpt,
    save_dir="./spiral_videos",
    save_name=None,
    n_frames=90,
    random_bg=False,
):
    ckpt = torch.load(ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    prompt = cfg["prompt"]["prompt"].replace(" ", "_")

    renderer = GaussianSplattingRenderer.load(None, ckpt).to("cuda")

    renderer.eval()

    c2ws = get_camera_path_fixed_elevation(
        n_frames, 1, np.mean(cfg["data"]["camera_distance"]), 45
    )
    c2ws = torch.from_numpy(c2ws).to(renderer.device)

    save_dir = Path(save_dir)
    if save_name is None:
        save_name = prompt
    else:
        save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    camera_info = CameraInfo.from_reso(512)
    rgb_frames = np.zeros([n_frames, 512, 512, 3], dtype=np.uint8)
    depth_frames = np.zeros([n_frames, 512, 512, 3], dtype=np.uint8)
    T_frames = np.zeros([n_frames, 512, 512, 1], dtype=np.uint8)

    if random_bg:
        bg = torch.rand(3, device="cuda", dtype=torch.float32)
        bg = repeat(bg, "c -> h w c", h=512, w=512)

    for idx, c2w in enumerate(c2ws):
        with torch.no_grad():
            out = renderer.render_one(c2w, camera_info, return_T=True, use_bg=False)
            if not random_bg:
                rgb = (out["rgb"].cpu().numpy() * 255).astype(np.uint8)
                depth = (
                    apply_depth_colormap(out["depth"], out["opacity"]).cpu().numpy()
                    * 255.0
                ).astype(np.uint8)
                T = 255 - (out["T"].cpu().numpy() * 255).astype(np.uint8)
                rgb_frames[idx] = rgb
                depth_frames[idx] = depth
                T_frames[idx] = T
            else:
                rgb = (out["rgb"] + out["T"] * bg).clamp(0.0, 1.0).cpu().numpy()
                rgb = (rgb * 255).astype(np.uint8)
                depth = (
                    apply_depth_colormap(out["depth"], out["opacity"]).cpu().numpy()
                    * 255.0
                ).astype(np.uint8)
                T = 255 - (out["T"].cpu().numpy() * 255).astype(np.uint8)
                rgb_frames[idx] = rgb
                depth_frames[idx] = depth
                T_frames[idx] = T

    frames = np.concatenate([rgb_frames, depth_frames], axis=2)
    write_video(save_dir / f"{save_name}.mp4", frames, fps=30)

    T_frames = np.concatenate([T_frames, T_frames], axis=2)
    frames_transparent_bg = np.concatenate([frames, T_frames], axis=3)
    # write_video(
    #     save_dir / f"{save_name}_transparent_bg.mov", frames_transparent_bg, fps=30
    # )
    # mimwrite(
    #     str(save_dir / f"{save_name}_transparent_bg.mov"),
    #     frames_transparent_bg,
    #     fps=30,
    #     quality=10,
    # )
    console.print(f"done for [red]{save_name.replace('_', ' ')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--save_dir", type=str, default="./spiral_videos")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--n_frames", type=int, default=90)
    parser.add_argument("--step", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_bg", action="store_true")

    opt = parser.parse_args()

    np.random.seed(opt.seed)

    if not opt.ckpt.endswith(".txt"):
        ckpt = Path(opt.ckpt)
        if not ckpt.exists():
            uid, time, day, prompt = str(ckpt).strip().split("|")

            ckpt_dir = Path(f"./checkpoints/{prompt}/{day}/{time}/ckpts/")
            if opt.step is None:
                files = ckpt_dir.glob("*.pt")
                latest_file = max(files, key=lambda x: x.stat().st_mtime)
                ckpt = latest_file
            else:
                ckpt = ckpt_dir / f"step_{opt.step}.pt"

        take_spiral_from_ckpt(
            ckpt, opt.save_dir, opt.save_name, opt.n_frames, opt.random_bg
        )
    else:
        with open(opt.ckpt, "r") as f:
            for line in tqdm(f):
                print(line)
                ckpt = Path(line.strip())
                if not ckpt.exists():
                    uid, time, day, prompt = str(ckpt).strip().split("|")

                    ckpt_dir = Path(f"./checkpoints/{prompt}/{day}/{time}/ckpts/")
                    if opt.step is None:
                        files = ckpt_dir.glob("*.pt")
                        try:
                            latest_file = max(files, key=lambda x: x.stat().st_mtime)
                        except:
                            continue
                        ckpt = latest_file
                    else:
                        ckpt = ckpt_dir / f"step_{opt.step}.pt"
                take_spiral_from_ckpt(
                    ckpt, opt.save_dir, opt.save_name, opt.n_frames, opt.random_bg
                )
