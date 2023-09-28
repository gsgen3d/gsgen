import gc
import requests
from . import BaseGuidance
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    IFPipeline,
    PNDMScheduler,
    DiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

from utils.typing import *
from utils.ops import perpendicular_component
from utils.misc import C
from rich.console import Console

console = Console()


class DeepFloydGuidance(BaseGuidance):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        if self.cfg.repeat_until_success:
            success = False
            while not success:
                try:
                    self.pipe = IFPipeline.from_pretrained(
                        self.cfg.pretrained_model_name_or_path,
                        text_encoder=None,
                        safety_checker=None,
                        watermarker=None,
                        feature_extractor=None,
                        requires_safety_checker=False,
                        variant="fp16" if self.cfg.half_precision_weights else None,
                        torch_dtype=self.weights_dtype,
                        cache_dir="./.cache",
                    ).to(self.device)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    console.print(".", end="")
                else:
                    success = True
                    break
        else:
            self.pipe = IFPipeline.from_pretrained(
                self.cfg.pretrained_model_name_or_path,
                text_encoder=None,
                safety_checker=None,
                watermarker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                variant="fp16" if self.cfg.half_precision_weights else None,
                torch_dtype=self.weights_dtype,
                cache_dir="./.cache",
            ).to(self.device)

        # self.pipe = IFPipeline.from_pretrained(
        #     self.cfg.pretrained_model_name_or_path,
        #     text_encoder=None,
        #     safety_checker=None,
        #     watermarker=None,
        #     feature_extractor=None,
        #     requires_safety_checker=False,
        #     variant="fp16" if self.cfg.half_precision_weights else None,
        #     torch_dtype=self.weights_dtype,
        # ).to(self.device)

        self.step = 0
        self.max_steps = self.cfg.max_steps
        self.unet = self.pipe.unet.eval()
        self.scheduler = self.pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.grad_clip_val = None
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.set_min_max_steps()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        self.use_upsample_model = self.cfg.get("use_upsample_model", False)
        if self.use_upsample_model:
            if self.cfg.repeat_until_success:
                ok = False
                while not ok:
                    try:
                        self.upsample_pipe = DiffusionPipeline.from_pretrained(
                            "DeepFloyd/IF-II-L-v1.0",
                            text_encoder=None,
                            variant="fp16",
                            torch_dtype=torch.float16,
                            safety_checker=None,
                            watermarker=None,
                            feature_extractor=None,
                            requires_safety_checker=False,
                            cache_dir="./.cache",
                        ).to(self.device)
                        ok = True
                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except requests.exceptions.ConnectionError:
                        console.print(".", end="")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self):
        min_step_percent = C(self.cfg.min_step_percent, self.step, self.max_steps)
        max_step_percent = C(self.cfg.max_step_percent, self.step, self.max_steps)
        self.min_t_step = int(self.num_train_timesteps * min_step_percent)
        self.max_t_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(self, latents, t, encoder_hidden_states):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]
        if use_perp_neg:
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 6, 64, 64)

            noise_pred_text, _ = noise_pred[:batch_size].split(3, dim=1)
            noise_pred_uncond, _ = noise_pred[batch_size : batch_size * 2].split(
                3, dim=1
            )
            noise_pred_neg, _ = noise_pred[batch_size * 2 :].split(3, dim=1)

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
        else:
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (2B, 6, 64, 64)

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return torch.cat([noise_pred, predicted_variance], dim=1)

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
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        assert rgb_as_latents == False, f"No latent space in {self.__class__.__name__}"
        rgb_BCHW = rgb_BCHW * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(
            rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
        )

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_t_step,
            self.max_t_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if prompt_embedding.use_perp_negative:
            # TODO: add code for perp negative
            pass
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_embedding.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distance, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 6, 64, 64)

            noise_pred_text, _ = noise_pred[:batch_size].split(3, dim=1)
            noise_pred_uncond, _ = noise_pred[batch_size : batch_size * 2].split(
                3, dim=1
            )
            noise_pred_neg, _ = noise_pred[batch_size * 2 :].split(3, dim=1)

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

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

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

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        loss_sds_each = 0.5 * F.mse_loss(latents, target, reduction="none").sum(
            dim=[1, 2, 3]
        )

        guidance_out = {
            "loss_sds": loss_sds,
            "loss_sds_each": loss_sds_each,
            "grad_norm": grad.norm(),
            "min_step": self.min_t_step,
            "max_step": self.max_t_step,
        }

        # if self.use_upsample_model:
        #     # TODO: finish this
        #     assert hasattr(self, "upsample_pipe"), "upsample_pipe not initialized"
        #     upsample_model_input = latents
        #     prompt_embeds, negative_embeds = text_embeddings.chunk(2)
        #     upsampled_image = self.upsample_pipe(
        #         image=upsample_model_input,
        #         prompt_embeds=prompt_embeds,
        #         negative_prompt_embeds=negative_embeds,
        #         output_type="pt",
        #         # generator=generator,
        #     ).images
        #     print(upsampled_image.shape)
        #     guidance_out["loss_upsample"] = None
        #     exit(0)

        if guidance_eval:
            guidance_eval_utils = {
                "use_perp_neg": prompt_embedding.use_perp_neg,
                "neg_guidance_weights": neg_guidance_weights,
                "text_embeddings": text_embeddings,
                "t_orig": t,
                "latents_noisy": latents_noisy,
                "noise_pred": torch.cat([noise_pred, predicted_variance], dim=1),
            }
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distance
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    def update(self, step):
        self.step = step
        self.set_min_max_steps()
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, self.step, self.max_steps)

    @torch.no_grad()
    def upsample_images(
        self,
        rgb,
        prompt_embedding,
        elevation,
        azimuth,
        camera_distance,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        rgb_BCHW = rgb_BCHW * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(
            rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
        )
        assert self.use_upsample_model
        text_embeddings = prompt_embedding.get_text_embedding(
            elevation, azimuth, camera_distance, self.cfg.use_view_dependent_prompt
        )
        prompt_embeds, negative_embeds = text_embeddings.chunk(2)
        upsampled_images = self.upsample_pipe(
            image=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            output_type="pt",
            # generator=generator,
        ).images

        upsampled_images = upsampled_images / 2.0 + 0.5
        return upsampled_images.moveaxis(1, -1).to(torch.float32)

    def delete_upsample_model(self):
        del self.upsample_pipe
        gc.collect()
        torch.cuda.empty_cache()
