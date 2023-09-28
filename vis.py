import time
import torch
from gs.gaussian_splatting import GaussianSplattingRenderer
from utils.viewer.viser_viewer import ViserViewer
from omegaconf import OmegaConf
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--port", type=int, default=8080)

    torch.set_default_device("cuda")

    opt = parser.parse_args()
    renderer = GaussianSplattingRenderer.load(None, opt.ckpt).to("cuda")
    renderer.eval()

    viewer_cfg = OmegaConf.create(
        {
            "device": "cuda",
            "viewer_port": opt.port,
        }
    )
    viewer = ViserViewer(viewer_cfg, train_mode=False)
    viewer.set_renderer(renderer)

    while True:
        viewer.update()
        time.sleep(1e-3)
