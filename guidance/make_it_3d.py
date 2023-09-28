from . import BaseGuidance
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionPipeline,
    PNDMScheduler,
)

from utils.typing import *
from utils.ops import perpendicular_component
from utils.misc import C
from rich.console import Console
import clip
import torchvision.transforms as T

console = Console()

from .stable_diffusion import StableDiffusionGuidance


class MakeIt3DGuidance(StableDiffusionGuidance):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.clip_model_name = self.cfg.get("clip_model_name", "ViT-B/16")
        self.clip, self.clip_process = clip.load(
            self.clip_model_name, device=self.device, jit=False
        )

        self.aug = T.Compose(
            [
                T.Resize((224, 224)),
                T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self.clip_to_sds = self.cfg.get("clip_to_sds", 0.4)
        self.clip.eval()

    def clip_encode_image(self, image):
        if isinstance(image, list):
            transformed_image = []
            for img in image:
                transformed_image.append(self.aug(img))
            image = torch.cat(transformed_image, dim=0)
            ret = self.clip.encode_image(image)
        else:
            ret = self.clip.encode_image(self.aug(image))
        return ret / ret.norm(dim=-1, keepdim=True)

    def clip_encode_text(self, text):
        ret = clip.tokenize(text).to(self.device)
        ret = self.clip.encode_text(ret)
        return ret / ret.norm(dim=-1, keepdim=True)

    def forward(
        self,
        rgb,
        prompt_embedding,
        elevation,
        azimuth,
        camera_distance,
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        noise_pred, noise, latents, latents_noisy, t = self.get_noise_pred(
            rgb,
            prompt_embedding,
            elevation,
            azimuth,
            camera_distance,
            rgb_as_latents,
            guidance_eval,
            **kwargs,
        )
        bs = rgb.shape[0]
        mask = (t / self.num_train_timesteps) < self.clip_to_sds
        num_clip_items = mask.sum().item()
        if num_clip_items == bs:
            loss_sds = 0.0
        else:
            loss_sds = self.get_sds_loss(
                latents[~mask], noise_pred[~mask], noise[~mask], t[~mask]
            )
        if num_clip_items == 0:
            loss_clip = 0.0
        else:
            loss_clip = self.get_clipd_loss(
                latents_noisy[mask],
                noise_pred[mask],
                t[mask],
                kwargs["image"],
                kwargs["text"],
            )

        return {
            "loss_sds": loss_sds,
            "loss_clip": loss_clip,
        }

    def get_noise_pred(
        self,
        rgb,
        prompt_embedding,
        elevation,
        azimuth,
        camera_distance,
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        bs = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        t = torch.randint(
            self.min_t_step,
            self.max_t_step + 1,
            [bs],
            dtype=torch.long,
            device=self.device,
        )

        if prompt_embedding.use_perp_negative:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_embedding.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distance, self.cfg.use_view_dependent_prompt
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:bs]
            noise_pred_uncond = noise_pred[bs : bs * 2]
            noise_pred_neg = noise_pred[bs * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
            noise_pred = noise_pred.to(torch.float32)
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_embedding.get_text_embedding(
                elevation, azimuth, camera_distance, self.cfg.use_view_dependent_prompt
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred, noise, latents, latents_noisy, t

    def get_sds_loss(self, latents, noise_pred, noise, t):
        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )
        grad = (noise_pred - noise) * w
        target = (latents - grad).detach()
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return loss_sds

    def get_clipd_loss(self, latents_noisy, noise_pred, t, image, text):
        bs = latents_noisy.shape[0]
        self.scheduler.set_timesteps(self.num_train_timesteps)
        de_latents = torch.zeros_like(noise_pred)
        for idx, (np, tt, ln) in enumerate(zip(noise_pred, t, latents_noisy)):
            de_latents[idx] = self.scheduler.step(np, tt, ln)["prev_sample"]
        # de_latents = self.scheduler.step(noise_pred, t, latents_noisy)["prev_sample"]
        imgs = self.decode_latents(de_latents)

        ref_image = image.unsqueeze(0).permute(0, 3, 1, 2)
        # clip_input = torch.cat([ref_image, imgs])
        img_embeds = self.clip_encode_image([ref_image, imgs])

        text_embeds = self.clip_encode_text(text).repeat(bs, 1)

        ref_embeds, img_embeds = img_embeds.split([1, bs], dim=0)

        ref_embeds = ref_embeds.expand_as(img_embeds)
        clip_image_loss = -(ref_embeds * img_embeds).sum(dim=-1).mean()

        clip_text_loss = -(text_embeds * img_embeds).sum(dim=-1).mean()

        return clip_image_loss + clip_text_loss

    def get_normal_clip_loss(self, image, ref_image, text):
        ref_image = ref_image.unsqueeze(0).permute(0, 3, 1, 2)
        bs = image.shape[0]
        image = image.permute(0, 3, 1, 2)
        # clip_input = torch.cat([ref_image, image], dim=0)
        img_embeds = self.clip_encode_image([ref_image, image])

        text_embeds = self.clip_encode_text(text)

        ref_embeds, img_embeds = img_embeds.split([1, bs], dim=0)

        ref_embeds = ref_embeds.expand_as(img_embeds)
        clip_image_loss = -(ref_embeds * img_embeds).sum(dim=-1).mean()

        clip_text_loss = -(text_embeds * img_embeds).sum(dim=-1).mean()

        return clip_image_loss + clip_text_loss
