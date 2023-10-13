import numpy as np
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import (
    create_pan_cameras,
    decode_latent_images,
    gif_widget,
    decode_latent_mesh,
)

# from shap_e.util.notebooks import decode_latent_mesh


def shap_e_generate_pcd_from_text(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    batch_size = 1
    guidance_scale = 15.0
    prompt = text

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    t = decode_latent_mesh(xm, latents[0]).tri_mesh()
    xyz = torch.from_numpy(t.verts)
    rgb = torch.from_numpy(np.stack([t.vertex_channels[x] for x in "RGB"], axis=1))

    pc = torch.cat([xyz, rgb], dim=-1)
    # pc[..., 3:] = pc[..., 3:].clamp(0.0, 255.0) / 255

    return pc
