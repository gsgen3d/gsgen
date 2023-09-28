import torch
import numpy as np
from torchtyping import TensorType
from utils.camera import CameraInfo
from typing import Tuple


@torch.no_grad()
def tile_culling_aabb_count(
    mean: TensorType["N", 2],
    cov: TensorType["N", 2, 2],
    tile_size: int,
    camera_info: CameraInfo,
    D: float,
) -> Tuple[int, TensorType["N", 2], TensorType["N", 2]]:
    centers = mean
    aabb_x = torch.sqrt(D * cov[:, 0, 0])
    aabb_y = torch.sqrt(D * cov[:, 1, 1])
    aabb_sidelength = torch.stack([aabb_x, aabb_y], dim=-1)
    aabb_topleft = centers - aabb_sidelength
    aabb_bottomright = centers + aabb_sidelength
    topleft_pixels = camera_info.camera_space_to_pixel_space(aabb_topleft)
    bottomright_pixels = camera_info.camera_space_to_pixel_space(aabb_bottomright)

    topleft_pixels[..., 0].clamp_(min=0, max=camera_info.w - 1)
    topleft_pixels[..., 1].clamp_(min=0, max=camera_info.h - 1)
    bottomright_pixels[..., 0].clamp_(min=0, max=camera_info.w - 1)
    bottomright_pixels[..., 1].clamp_(min=0, max=camera_info.h - 1)

    topleft_pixels = torch.div(topleft_pixels, tile_size, rounding_mode="floor")
    bottomright_pixels = torch.div(bottomright_pixels, tile_size, rounding_mode="floor")

    N_with_dub = (
        torch.prod(bottomright_pixels - topleft_pixels + 1, dim=-1).sum().item()
    )

    return N_with_dub, topleft_pixels, bottomright_pixels
