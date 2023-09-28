from pathlib import Path
from PIL import Image
import numpy as np
import torch

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config


def point_e_generate_pcd_from_text(text, num_points=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Point-E on device:", device)

    print("creating base model...")
    base_name = "base40M-textvec"
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print("creating upsample model...")
    upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

    print("downloading base checkpoint...")
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print("downloading upsampler checkpoint...")
    upsampler_model.load_state_dict(load_checkpoint("upsample", device))
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, num_points - 1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=("texts", ""),  # Do not condition the upsampler at all
    )
    # Set a prompt to condition on.
    prompt = text

    # Produce a sample from the model.
    samples = None
    for x in sampler.sample_batch_progressive(
        batch_size=1, model_kwargs=dict(texts=[prompt])
    ):
        samples = x
    pc = sampler.output_to_point_clouds(samples)[0]

    xyz = torch.from_numpy(pc.coords).to(torch.float32)
    rgb = torch.from_numpy(
        np.stack([pc.channels[c] for c in ["R", "G", "B"]], axis=-1)
    ).to(torch.float32)
    # breakpoint()

    pc = torch.cat([xyz, rgb], dim=-1)
    # pc[..., 3:] = pc[..., 3:].clamp(0.0, 255.0) / 255

    return pc


def point_e_generate_pcd_from_image(image, num_points=4096, base_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("creating base model...")
    if base_name is None:
        base_name = "base1B"  # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print("creating upsample model...")
    upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

    print("downloading base checkpoint...")
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print("downloading upsampler checkpoint...")
    upsampler_model.load_state_dict(load_checkpoint("upsample", device))

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, num_points - 1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0, 3.0],
    )

    samples = None

    if isinstance(image, torch.Tensor):
        image = (
            (image.detach().cpu().squeeze().numpy() * 255)
            .astype(np.uint8)
            .clip(min=0, max=255)
        )
        image = Image.fromarray(image)
    elif isinstance(image, str) or isinstance(image, Path):
        image = Image.open(image)
    else:
        raise TypeError("image must be a torch.Tensor or a path to an image file")

    for x in sampler.sample_batch_progressive(
        batch_size=1, model_kwargs=dict(images=[image])
    ):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    xyz = torch.from_numpy(pc.coords).to(torch.float32)
    rgb = torch.from_numpy(
        np.stack([pc.channels[c] for c in ["R", "G", "B"]], axis=-1)
    ).to(torch.float32)
    # breakpoint()

    pc = torch.cat([xyz, rgb], dim=-1)

    return pc
