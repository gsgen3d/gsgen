import argparse
import numpy as np
import torch
import struct
from pathlib import Path
from gs.gaussian_splatting import GaussianSplattingRenderer
from utils.ops import K_nearest_neighbors
from utils.ckpt import get_ckpt_path
from plyfile import PlyData, PlyElement

from rich.console import Console

console = Console()


def get_densify_val_grid(
    renderer: GaussianSplattingRenderer,
    L: float = -1.0,
    reso: int = 256,
    K: int = 3,
):
    if L < 0.0:
        L = renderer.mean.abs().max().item() * 1.1
    x = torch.linspace(-L, L, reso)
    y = torch.linspace(-L, L, reso)
    z = torch.linspace(-L, L, reso)

    x, y, z = torch.meshgrid(x, y, z)
    grid = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)

    mean = renderer.mean

    _, nn_idx = K_nearest_neighbors(grid, mean, K=K)

    densify_val_grid = torch.zeros_like(grid[..., :1])


def to_ply(ckpt_path, save_dir):
    ckpt_path = get_ckpt_path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    prompt = cfg["prompt"]["prompt"].replace(" ", "_")

    if "params" in ckpt:
        ckpt = ckpt["params"]

    # align with the official implmentation
    fields = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "red",
        "green",
        "blue",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]

    with torch.no_grad():
        pos = ckpt["mean"].numpy()
        normals = np.zeros_like(pos)
        rgb = ckpt["color"].numpy() * 255.0
        opacity = ckpt["alpha"].numpy()[..., None]
        svec = ckpt["svec"].numpy()
        qvec = ckpt["qvec"].numpy()

    dtype_full = [(attribute, "f4") for attribute in fields]

    elements = np.empty(pos.shape[0], dtype=dtype_full)
    attributes = np.concatenate((pos, normals, rgb, opacity, svec, qvec), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")

    save_dir = Path(save_dir) / "ply"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    PlyData([el]).write(str(save_dir / f"{prompt}.ply"))

    console.print(f"[red]save to {str(save_dir / f'{prompt}.ply')}[/red]")


def to_splat(ckpt_path, save_dir):
    """convert checkpoint to splat format defined by web viewer

    Args:
        ckpt_path (Union[str, Path]): checkpoint path or uid string
        save_dir (str) save directory
    """
    ckpt_path = get_ckpt_path(ckpt_path)
    if ckpt_path is None:
        console.print(f"[red]ckpt not found: {ckpt_path}[/red]")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    prompt = cfg["prompt"]["prompt"].replace(" ", "_")

    if "params" in ckpt:
        ckpt = ckpt["params"]

    # align with the official implmentation
    fields = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "red",
        "green",
        "blue",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]

    with torch.no_grad():
        pos = ckpt["mean"].numpy()
        rgb = (
            (torch.sigmoid(ckpt["color"]).numpy() * 255.0).astype(np.uint8).clip(0, 255)
        )
        opacity = (
            (torch.sigmoid(ckpt["alpha"]).numpy()[..., None] * 255.0)
            .astype(np.uint8)
            .clip(0, 255)
        )
        svec = np.exp(ckpt["svec"].numpy())
        qvec = ckpt["qvec"].numpy()
        qvec = qvec / np.linalg.norm(qvec, axis=1, keepdims=True)
        qvec = qvec * 128 + 128
        qvec = qvec.astype(np.uint8).clip(0, 255)

    n = pos.shape[0]

    volume = np.prod(svec, axis=1) * opacity[..., 0]
    index = list(range(n))
    index = sorted(index, key=lambda i: volume[i], reverse=True)

    save_dir = Path(save_dir) / "splat"
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    filename = str(save_dir / f"{prompt}.splat")
    with open(filename, "wb") as f:
        for i in index:
            f.write(struct.pack("fff", pos[i, 0], pos[i, 1], pos[i, 2]))
            f.write(struct.pack("fff", svec[i, 0], svec[i, 1], svec[i, 2]))
            f.write(struct.pack("BBBB", rgb[i, 0], rgb[i, 1], rgb[i, 2], opacity[i, 0]))
            f.write(struct.pack("BBBB", qvec[i, 0], qvec[i, 1], qvec[i, 2], qvec[i, 3]))

    console.print(f"[red]save to {str(save_dir / f'{prompt}.splat')}[/red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--type", type=str, default="ply")
    parser.add_argument("--save_dir", type=str, default="./exports")

    opt = parser.parse_args()
    func = None

    if opt.type == "ply":
        func = to_ply
    elif opt.type == "splat":
        func = to_splat
    else:
        raise NotImplementedError(f"Unknown export type: {opt.type}")

    if opt.ckpt.endswith(".txt"):
        with open(opt.ckpt, "r") as f:
            for line in f:
                ckpt_path = line.strip()
                func(ckpt_path, opt.save_dir)
    else:
        func(opt.ckpt, opt.save_dir)

    # if opt.type == "ply":
    #     to_ply(opt.ckpt, opt.save_dir)
    # elif opt.type == "splat":
    #     to_splat(opt.ckpt, opt.save_dir)
    # else:
    #     raise NotImplementedError(f"Unknown export type: {opt.type}")
