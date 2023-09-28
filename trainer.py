import os
import gc
import numpy as np
import datetime
import warnings
from pathlib import Path
import torch
from tqdm import tqdm
from PIL import Image, ImageFilter
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from einops import repeat
from omegaconf import OmegaConf
from data import CameraPoseProvider, SingleViewCameraPoseProvider
from gs.gaussian_splatting import GaussianSplattingRenderer
from utils.misc import (
    to_primitive,
    C,
    step_check,
    stack_dicts,
    get_file_list,
    dict_to_device,
    dump_config,
    huggingface_online,
    huggingface_offline,
    get_current_cmd,
    get_dict_slice,
    seed_everything,
)
from utils.transforms import qvec2rotmat_batched
from utils.ops import binary_cross_entropy
from utils.initialize import base_initialize, initialize
from utils.dpt import DPT
from utils.spiral import (
    get_camera_path_fixed_elevation,
    get_random_pose_fixed_elevation,
)
from utils.colormaps import apply_float_colormap, apply_depth_colormap
from utils.wandb import get_num_runs
from utils.loss import depth_loss, get_image_loss
from guidance import get_guidance
from prompt import get_prompt_processor
import wandb
import shutil
import imageio
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from torchmetrics import PearsonCorrCoef

console = Console()


def convert_to_image(outs):
    outs["depth"] = apply_depth_colormap(outs["depth"], outs["opacity"])
    outs["opacity"] = apply_float_colormap(outs["opacity"])

    final = torch.cat(list(outs.values()), dim=-2)


class Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.step = 0
        self.max_steps = cfg.max_steps
        self.mode = cfg.get("mode", "text_to_3d")

        disable_warnings = self.cfg.get("disable_warnings", False)
        if disable_warnings:
            console.print(f"[red]Ignore All Warnings!!!")
            warnings.simplefilter("ignore")

        try:
            torch.set_default_device(cfg.device)
            torch.set_default_dtype(torch.float32)
        except AttributeError:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # FIXME setting offline flags should be done before importing transformers
        if self.cfg.huggingface_offline:
            huggingface_offline()
        else:
            huggingface_online()

        prompt = (
            self.cfg.prompt.prompt.strip().replace(" ", "_").lower()[:64]
        )  # length limited by wandb
        day_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        hms_timestamp = datetime.datetime.now().strftime("%H%M%S")
        timestamp = f"{hms_timestamp}|{day_timestamp}"
        num_runs = get_num_runs("gsgen")
        uid = f"{num_runs}|{timestamp}|{prompt}"
        tags = [day_timestamp, prompt, self.cfg.guidance.type, self.mode]
        notes = self.cfg.notes
        self.timestamp = timestamp

        self.depth_estimator = None
        if cfg.estimators.depth.enabled:
            self.depth_estimator = DPT(device=cfg.device, mode="depth")
            self.pearson = PearsonCorrCoef().to(cfg.device)

        if cfg.estimators.normal.enabled:
            self.normal_estimator = DPT(device=cfg.device, mode="normal")

        if self.mode == "text_to_3d":
            self.dataset = CameraPoseProvider(cfg.data)
        elif self.mode == "image_to_3d":
            self.dataset = SingleViewCameraPoseProvider(cfg.data)
            self.text_prompt = self.cfg.prompt.prompt
        self.loader = iter(
            DataLoader(
                self.dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=self.dataset.collate,
                num_workers=0,
            )
        )

        if self.mode == "image_to_3d":
            assert "image" in self.cfg, "image should be provided in image_to_3d mode"
            assert (
                self.depth_estimator is not None
            ), "depth estimator should be provided"
            image = Path(self.cfg.image)
            assert image.exists(), f"{image} not exists"
            image = Image.open(image)
            if self.cfg.get("image_blur", False):
                image = image.filter(ImageFilter.GaussianBlur(radius=3))
            self.image = ToTensor()(image).moveaxis(0, -1)
            self.mask = self.image[..., 3] > 0.0
            self.image = self.image[..., :3].to(self.cfg.device)
            self.depth_map = self.depth_estimator(self.image[None, ...])
            # NOTE: I found this important
            # TODO: validate this is correct way to normalize depth
            self.depth_map = (
                (self.depth_map - self.depth_map[0][self.mask].mean())
                * self.cfg.get("depth_scale", 100.0)
                # * self.dataset.get_reso
                # / 256
                + self.dataset.original_camera_distance
            )
            initial_values = initialize(
                cfg.init,
                image=self.image,
                depth_map=self.depth_map,
                mask=self.mask,
                c2w=self.dataset.original_out["c2w"],
                camera_info=self.dataset.original_out["camera_info"],
            )

            self.image_loss_fn = get_image_loss(0.2, "l2")
        elif self.mode == "text_to_3d":
            initial_values = initialize(cfg.init)
        # initial_values = base_initialize(cfg.init)
        self.renderer = GaussianSplattingRenderer(
            cfg.renderer, initial_values=initial_values
        ).to(cfg.device)
        self.renderer.setup_lr(cfg.lr)
        self.renderer.set_optimizer(cfg.optimizer)

        if self.cfg.aux.enabled:
            self.aux_guidance = get_guidance(cfg.aux)
            self.aux_guidance.set_text(
                self.cfg.aux.get("prompt", self.cfg.prompt.prompt)
            )

        self.guidance = get_guidance(cfg.guidance)
        if self.cfg.guidance.get("keep_complete_pipeline", False):
            self.prompt_processor = get_prompt_processor(
                cfg.prompt, guidance_model=self.guidance
            )
        else:
            self.prompt_processor = get_prompt_processor(cfg.prompt)

        self.prompt_processor.cleanup()
        gc.collect()
        torch.cuda.empty_cache()

        self.save_dir = Path(f"./checkpoints/{prompt}/{day_timestamp}/{hms_timestamp}")
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(f"./logs/{prompt}/{day_timestamp}/{hms_timestamp}")
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir = self.save_dir / "eval"
        if not self.eval_dir.exists():
            self.eval_dir.mkdir(parents=True, exist_ok=True)

        wandb.tensorboard.patch(root_logdir=str(self.log_dir))

        overrided_group = self.cfg.get("group", prompt)
        addtional_tags = self.cfg.get("tags", [])
        tags = tags + addtional_tags

        if cfg.wandb:
            wandb.init(
                project="gsgen",
                name=uid,
                config=to_primitive(cfg),
                sync_tensorboard=True,
                # magic=True,
                save_code=True,
                group=overrided_group,
                notes=notes,
                tags=tags,
            )
            wandb.watch(
                self.renderer,
                log="all",
                log_freq=100,
            )

        self.writer = SummaryWriter(str(self.log_dir))

        cmd = get_current_cmd()
        self.writer.add_text("cmd", cmd, 0)
        self.save_code_snapshot()
        self.start = 0
        self.last_out = None

        console.print(f"[red]UID: {uid} started")

    @property
    def optimizer(self):
        return self.renderer.optimizer

    @classmethod
    def load(cls, ckpt, override_cfg=None):
        if not isinstance(ckpt, dict):
            ckpt = torch.load(ckpt, map_location="cpu")

        step = ckpt["step"]
        cfg = OmegaConf.create(ckpt["cfg"])
        if override_cfg is not None:
            cfg.update(override_cfg)

        trainer = cls(cfg)
        trainer.renderer = GaussianSplattingRenderer.load(
            cfg.renderer, ckpt["params"]
        ).to(cfg.device)

        trainer.renderer.setup_lr(cfg.lr)
        trainer.renderer.set_optimizer(cfg.optimizer)

        trainer.step = step
        trainer.update(step)

        return trainer

    def save(self):
        params = self.renderer.get_params_for_save()
        cfg = to_primitive(self.cfg)
        state = {
            "params": params,
            "cfg": cfg,
            "step": self.step,
        }
        save_dir = self.save_dir / "ckpts"
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, self.save_dir / "ckpts" / f"step_{self.step}.pt")

    def save_code_snapshot(self):
        # learned from threestudio
        self.code_dir = self.save_dir / "code"
        if not self.code_dir.exists():
            self.code_dir.mkdir(parents=True, exist_ok=True)

        files = get_file_list()
        for f in files:
            dst = self.code_dir / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, str(dst))

        config_dir = self.save_dir / "config" / "parsed.yaml"
        if not config_dir.parent.exists():
            config_dir.parent.mkdir(parents=True, exist_ok=True)
        dump_config(str(config_dir), self.cfg)

    def update(self, step):
        self.dataset.update(step)
        self.renderer.update(step)
        self.guidance.update(step)
        self.prompt_processor.update(step)

    def train_step(self):
        self.train()
        batch = next(self.loader)
        out = self.renderer(batch, self.cfg.use_bg, self.cfg.rgb_only)
        prompt_embeddings = self.prompt_processor()
        guidance_out = self.guidance(
            out["rgb"],
            prompt_embeddings,
            elevation=batch["elevation"],
            azimuth=batch["azimuth"],
            camera_distance=batch["camera_distance"],
            c2w=batch["c2w"],
            rgb_as_latents=False,
        )
        loss = 0.0
        if "loss_sds" in guidance_out.keys():
            loss += (
                C(self.cfg.loss.sds, self.step, self.max_steps)
                * guidance_out["loss_sds"]
            )
            self.writer.add_scalar(
                "loss_weights/sds",
                C(self.cfg.loss.sds, self.step, self.max_steps),
                self.step,
            )

            self.writer.add_scalar("loss/sds", guidance_out["loss_sds"], self.step)

        if "loss_vsd" in guidance_out.keys():
            loss += (
                C(self.cfg.loss.vsd, self.step, self.max_steps)
                * guidance_out["loss_vsd"]
            )
            self.writer.add_scalar(
                "loss_weights/vsd",
                C(self.cfg.loss.vsd, self.step, self.max_steps),
                self.step,
            )

            self.writer.add_scalar("loss/vsd", guidance_out["loss_vsd"], self.step)

        if "loss_lora" in guidance_out.keys():
            loss += (
                C(self.cfg.loss.lora, self.step, self.max_steps)
                * guidance_out["loss_lora"]
            )
            self.writer.add_scalar(
                "loss_weights/lora",
                C(self.cfg.loss.lora, self.step, self.max_steps),
                self.step,
            )

            self.writer.add_scalar("loss/lora", guidance_out["loss_lora"], self.step)

        if self.cfg.loss.sparsity > 0.0:
            assert (
                "opacity" in out
            ), "opacity not in output, should turn off the `rgb_only` flag"
            sparsity_loss = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.writer.add_scalar("loss/sparsity", sparsity_loss, self.step)
            loss += C(self.cfg.loss.sparsity, self.step, self.max_steps) * sparsity_loss
            self.writer.add_scalar(
                "loss_weights/sparsity",
                C(self.cfg.loss.sparsity, self.step, self.max_steps),
            )

        if self.cfg.loss.opague > 0.0:
            assert (
                "opacity" in out
            ), "opacity not in output, should turn off the `rgb_only` flag"
            opacity_clamped = out["opacity"].clamp(1e-3, 1.0 - 1e-3)
            opacity_loss = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.writer.add_scalar("loss/opague", opacity_loss, self.step)
            loss += C(self.cfg.loss.opague, self.step, self.max_steps) * opacity_loss
            self.writer.add_scalar(
                "loss_weights/opague",
                C(self.cfg.loss.opague, self.step, self.max_steps),
            )

        if self.cfg.loss.z_var > 0:
            assert (
                "z_var" in out
            ), "z_var not in output, should turn on the `z_var` flag"
            opacity_clamped = out["opacity"].clamp(1e-3, 1.0 - 1e-3)
            z_var_loss = (
                out["z_var"] / opacity_clamped * (opacity_clamped > 0.5)
            ).mean()
            self.writer.add_scalar("loss/z_var", z_var_loss, self.step)
            loss += C(self.cfg.loss.z_var, self.step, self.max_steps) * z_var_loss
            self.writer.add_scalar(
                "loss_weights/z_var", C(self.cfg.loss.z_var, self.step, self.max_steps)
            )

        self.writer.add_scalar("loss/total", loss, self.step)

        # self.optimizer.zero_grad()
        loss += self.estimator_loss_step(out)
        loss = loss / self.cfg.grad_accum
        loss.backward()
        # self.optimizer.step()
        # self.renderer.post_backward()

        with torch.no_grad():
            if step_check(self.step, self.cfg.log_period, run_at_zero=True):
                out = dict_to_device(out, "cpu")
                train_image_pth = self.eval_dir / "train"
                if not train_image_pth.exists():
                    train_image_pth.mkdir(parents=True, exist_ok=True)
                if "depth" in out.keys():
                    assert "opacity" in out.keys()
                    out["depth"] = apply_depth_colormap(out["depth"], out["opacity"])
                    out["opacity"] = apply_float_colormap(out["opacity"])

                if "z_var" in out.keys():
                    out["z_var"] = (
                        out["z_var"] / out["opacity"] * (out["opacity"] > 0.5)
                    )
                    out["z_var"] = apply_float_colormap(
                        out["z_var"] / out["z_var"].max()
                    )

                final = (
                    torch.cat(list(out.values()), dim=-2).clamp(0, 1).cpu().numpy()
                    * 255.0
                ).astype(np.uint8)[-1]
                imageio.imwrite(str(train_image_pth / f"{self.step}.png"), final)
                self.writer.add_image(
                    "train/image", final, self.step, dataformats="HWC"
                )

        # return loss.item()
        return loss.item()

    def estimator_loss_step(self, out):
        loss = 0.0
        if self.cfg.estimators.depth.enabled:
            depth_estimated = self.depth_estimator(out["rgb"])
            assert (
                "depth" in out.keys()
            ), "depth should be rendered when using depth estimator loss"
            # should add a mask here to filter out the background
            depth_estimate_loss = depth_loss(
                self.pearson, depth_estimated, out["depth"]
            )
            self.writer.add_scalar("loss/depth", depth_estimate_loss, self.step)
            depth_loss_weight = C(
                self.cfg.estimators.depth.value, self.step, self.max_steps
            )
            self.writer.add_scalar("loss_weights/depth", depth_loss_weight, self.step)
            loss += depth_loss_weight * depth_estimate_loss

        if self.cfg.estimators.normal.enabled:
            normal_estimated = self.normal_estimator(out["rgb"])

            assert (
                "normal" in out.keys()
            ), "normal should be rendered when using normal estimator loss"
            normal_estimator_loss = F.mse_loss(out["normal"], normal_estimated)
            self.writer.add_scalar("estimator_loss/normal", normal_estimator_loss)

            loss += (
                C(self.cfg.estimators.normal.value, self.step, self.max_steps)
                * normal_estimator_loss
            )

        return loss

    def aux_guidance_step(self):
        if self.cfg.aux.enabled:
            aux_guidance_loss = self.aux_guidance(self.renderer)
            self.writer.add_scalar("loss/aux_guidance", aux_guidance_loss, self.step)
            loss = (
                C(self.cfg.loss.aux_guidance, self.step, self.max_steps)
                * aux_guidance_loss
            )
            loss.backward()

    def auxiliary_loss_step(self):
        loss = self.renderer.auxiliary_loss(self.step, self.writer)
        if loss.requires_grad:
            loss.backward()

    @torch.no_grad()
    def eval_image_step(self):
        self.eval()

        eval_image_path = self.eval_dir / "image"
        if not eval_image_path.exists():
            eval_image_path.mkdir(parents=True, exist_ok=True)

        if self.mode == "text_to_3d":
            c2w = get_random_pose_fixed_elevation(
                np.mean(self.dataset.camera_distance),
                self.cfg.eval.elevation,
            )
            camera_info = self.dataset.get_default_camera_info()
            c2w = torch.from_numpy(c2w)
        elif self.mode == "image_to_3d":
            c2w = self.dataset.original_out["c2w"]
            camera_info = self.dataset.original_out["camera_info"]
        else:
            raise NotImplementedError
        c2w = c2w.to(self.renderer.device)

        eval_upsample = self.cfg.get("eval_upsample", 1)
        camera_info.upsample(eval_upsample)

        out = self.renderer.render_one(
            c2w, camera_info, use_bg=self.cfg.use_bg, rgb_only=self.cfg.rgb_only
        )
        out = dict_to_device(out, "cpu")
        if "depth" in out.keys():
            assert "opacity" in out.keys()
            out["depth"] = apply_depth_colormap(out["depth"], out["opacity"])
            out["opacity"] = apply_float_colormap(out["opacity"])

        if "z_var" in out.keys():
            out["z_var"] = out["z_var"] / out["opacity"] * (out["opacity"] > 0.5)
            out["z_var"] = apply_float_colormap(out["z_var"] / out["z_var"].max())

        final = (torch.cat(list(out.values()), dim=-2).cpu().numpy() * 255.0).astype(
            np.uint8
        )
        imageio.imwrite(str(eval_image_path / f"{self.step}.png"), final)
        self.writer.add_image("eval/image", final, self.step, dataformats="HWC")

        self.train()

    @torch.no_grad()
    def eval_video_step(self):
        self.eval()

        eval_video_path = self.eval_dir / "video"
        if not eval_video_path.exists():
            eval_video_path.mkdir(parents=True, exist_ok=True)

        c2ws = get_camera_path_fixed_elevation(
            self.cfg.eval.n_frames,
            self.cfg.eval.n_circles,
            np.mean(self.dataset.camera_distance),
            self.cfg.eval.elevation,
        )
        c2ws = torch.from_numpy(c2ws).to(self.renderer.device)
        camera_info = self.dataset.get_default_camera_info()
        eval_upsample = self.cfg.get("eval_upsample", 1)
        camera_info.upsample(eval_upsample)

        outs = []
        use_bg = True
        if self.renderer.bg.type == "random":
            use_bg == False
        with torch.no_grad():
            for c2w in c2ws:
                out = self.renderer.render_one(
                    c2w, camera_info, use_bg=use_bg, rgb_only=self.cfg.rgb_only
                )
                outs.append(dict_to_device(out, "cpu"))

        outs = stack_dicts(outs)

        if "depth" in outs.keys():
            assert "opacity" in outs.keys()
            outs["depth"] = apply_depth_colormap(outs["depth"], outs["opacity"])
            outs["opacity"] = apply_float_colormap(outs["opacity"])

        if "z_var" in out.keys():
            outs["z_var"] = outs["z_var"] / outs["opacity"] * (outs["opacity"] > 0.5)
            outs["z_var"] = apply_float_colormap(outs["z_var"] / outs["z_var"].max())

        save_format = self.cfg.eval.save_format
        assert save_format in ["gif", "mp4"]

        final = torch.cat(list(outs.values()), dim=-2)  # concat at W
        imageio.mimwrite(
            str(eval_video_path / f"{self.step}.{save_format}"),
            (final.cpu().numpy() * 255).astype(np.uint8),
        )

        final = final.moveaxis(-1, -3)[None, ...]  # THWC -> TCHW
        self.writer.add_video(
            "eval/spiral",
            final,
        )
        self.train()

    def train_loop(self):
        self.train()

        with tqdm(total=self.max_steps - self.start) as pbar:
            for s in range(self.start, self.max_steps):
                self.step = s
                self.update(self.step)
                self.guidance.log(self.writer, s)
                self.dataset.log(self.writer, s)
                loss = 0.0

                for _ in range(self.cfg.grad_accum):
                    if self.mode == "text_to_3d":
                        loss += self.train_step()
                    elif self.mode == "image_to_3d":
                        loss += self.train_step_sit3d()
                    else:
                        raise NotImplementedError

                self.aux_guidance_step()
                self.auxiliary_loss_step()
                # loss += self.renderer.auxiliary_loss(s, self.writer)

                self.optimizer.step()
                self.renderer.post_backward()

                if step_check(s, self.cfg.log_period):
                    self.renderer.log(self.writer, s)

                if step_check(s, self.cfg.eval.image_period):
                    self.eval_image_step()

                if step_check(s, self.cfg.eval.video_period, True):
                    self.eval_video_step()

                if step_check(s, self.cfg.save_period, True):
                    self.save()

                ## TODO: add densify code
                self.renderer.densify(s)
                self.renderer.prune(s)
                self.optimizer.zero_grad()

                pbar.set_description(f"{self.timestamp}|Iter: {s}/{self.max_steps}")
                pbar.set_postfix(loss=f"{loss:.4f}")
                pbar.update(1)

    def train_step_sit3d(self):
        self.train()
        batch = next(self.loader)
        out = self.renderer(batch, self.cfg.use_bg, self.cfg.rgb_only)
        prompt_embeddings = self.prompt_processor()
        guidance_out = self.guidance(
            out["rgb"],
            prompt_embeddings,
            rgb_as_latents=False,
            elevation=batch["elevation"],
            azimuth=batch["azimuth"],
            camera_distance=batch["camera_distance"],
            image=self.image,
            text=self.text_prompt,
        )
        is_original_view_mask = batch["is_original_view"]

        loss = 0.0
        num_original_views = torch.sum(is_original_view_mask).item()
        bs = self.cfg.batch_size
        # sds_loss
        if "loss_sds" in guidance_out.keys():
            loss += (
                C(self.cfg.loss.sds, self.step, self.max_steps)
                * guidance_out["loss_sds"]
            )
            self.writer.add_scalar("loss/sds", guidance_out["loss_sds"], self.step)
        elif "loss_clip" in guidance_out.keys():
            loss += (
                C(self.cfg.loss.clip, self.step, self.max_steps)
                * guidance_out["loss_clip"]
            )
            self.writer.add_scalar("loss/clip", guidance_out["loss_clip"], self.step)
        else:
            raise ValueError("No guidance loss is provided")

        # image loss
        _, h, w, _ = out["rgb"].shape
        image = F.interpolate(
            self.image.moveaxis(-1, 0)[None, ...],
            (h, w),
            mode="bilinear",
            align_corners=False,
        )[0].moveaxis(0, -1)
        # print(self.depth_map.shape)
        depth = F.interpolate(
            self.depth_map[0].moveaxis(-1, 0)[None, ...],
            (h, w),
            mode="bilinear",
            align_corners=False,
        )[0].moveaxis(0, -1)
        if num_original_views > 0:
            image_loss = self.image_loss_fn(
                out["rgb"][is_original_view_mask],
                repeat(image, "h w c -> b h w c", b=num_original_views),
            )
            loss += C(self.cfg.loss.image, self.step, self.max_steps) * image_loss
            self.writer.add_scalar("loss/image", image_loss, self.step)

            depth_loss_val = depth_loss(
                self.pearson,
                out["depth"],
                # repeat(self.depth_map, "h w c -> b h w c", b=num_original_views),
                depth.repeat(num_original_views, 1, 1, 1),
            )
            loss += C(self.cfg.loss.depth, self.step, self.max_steps) * depth_loss_val
            self.writer.add_scalar("loss/depth", depth_loss_val, self.step)

        if num_original_views < bs:
            loss += self.guidance.get_normal_clip_loss(
                out["rgb"][~is_original_view_mask], self.image, self.text_prompt
            ) * C(self.cfg.loss.ref, self.step, self.max_steps)

        self.writer.add_scalar("loss/total", loss, self.step)

        # self.optimizer.zero_grad()
        # loss += self.estimator_loss_step(out)
        loss = loss / self.cfg.grad_accum
        loss.backward()
        # self.optimizer.step()
        # self.renderer.post_backward()

        with torch.no_grad():
            if step_check(self.step, self.cfg.log_period, run_at_zero=True):
                out = dict_to_device(out, "cpu")
                train_image_pth = self.eval_dir / "train"
                if not train_image_pth.exists():
                    train_image_pth.mkdir(parents=True, exist_ok=True)
                if "depth" in out.keys():
                    assert "opacity" in out.keys()
                    out["depth"] = apply_depth_colormap(out["depth"], out["opacity"])
                    out["opacity"] = apply_float_colormap(out["opacity"])

                if "z_var" in out.keys():
                    out["z_var"] = (
                        out["z_var"] / out["opacity"] * (out["opacity"] > 0.5)
                    )
                    out["z_var"] = apply_float_colormap(
                        out["z_var"] / out["z_var"].max()
                    )

                final = (
                    torch.cat(list(out.values()), dim=-2).clamp(0, 1).cpu().numpy()
                    * 255.0
                ).astype(np.uint8)[-1]
                imageio.imwrite(str(train_image_pth / f"{self.step}.png"), final)
                self.writer.add_image(
                    "train/image", final, self.step, dataformats="HWC"
                )

        # return loss.item()
        return loss.item()

    def tune_with_upsample_model(self):
        # total = self.t
        seed_everything(42)
        total = self.cfg.upsample_tune.num_poses
        self.image_loss_fn = get_image_loss(0.2, "l2")
        # self.image_loss_fn = F.mse_loss
        bs = self.cfg.upsample_tune.batch_size
        total = int(total / bs) * bs

        self.dataset = CameraPoseProvider(self.cfg.data)
        self.dataset.update(self.max_steps)
        print(self.dataset.get_elevation_bound)
        self.dataset.set_reso(64)
        if self.cfg.upsample_tune.get("uniform", False):
            console.print("[red]Using randomly sampled batch")
            all_data = self.dataset.get_batch(total)
        else:
            console.print("[red]Using uniformly sampled batch")
            all_data = self.dataset.get_uniform_batch(total)
        upsampled_images = []

        self.renderer.eval()

        cache_uid = f"{self.cfg.prompt.prompt.replace(' ', '')}_{self.cfg.ckpt.replace('/', '')}_{self.cfg.upsample_tune.num_poses}"

        cache_tmp_file = Path(f"./tmp/{cache_uid}.pt")
        if not self.cfg.upsample_tune.use_cache or not cache_tmp_file.exists():
            console.print(f"[green]no cache found, will save to {str(cache_tmp_file)}")
            for i in range(0, total, bs):
                batch = get_dict_slice(all_data, i, i + bs)
                rgb = self.renderer(batch, rgb_only=True)["rgb"]
                image_batch = self.guidance.upsample_images(
                    rgb=rgb,
                    prompt_embedding=self.prompt_processor(),
                    elevation=batch["elevation"],
                    azimuth=batch["azimuth"],
                    camera_distance=batch["camera_distance"],
                )
                upsampled_images.append(image_batch.cpu())

            self.guidance.delete_upsample_model()
            upsampled_images = torch.cat(upsampled_images, dim=0)
            self.renderer.train()
            torch.save(upsampled_images, cache_tmp_file)
        else:
            console.print("[green]load from cache")
            upsampled_images = torch.load(cache_tmp_file, map_location="cpu")

        reso = self.cfg.upsample_tune.reso
        self.dataset.set_reso(reso)  # actaully not used
        for cam_info in all_data["camera_info"]:
            cam_info.set_reso(reso)

        epoch = self.cfg.upsample_tune.epoch
        console.print(
            f"Step: {self.step}, start tuning with upsampling model for {epoch} epoch"
        )

        self.update(self.cfg.max_steps)
        if hasattr(self.cfg.upsample_tune, "lr"):
            self.renderer.setup_lr(self.cfg.upsample_tune.lr)
            self.renderer.set_optimizer(self.cfg.upsample_tune.optimizer)

        if self.cfg.upsample_tune.get("densify", False):
            num_densified = self.renderer.densify_by_compatness(3)
            self.renderer.reset_densify_info()
            console.print(f"[red]densify enabled, {num_densified} densified")

        if self.cfg.upsample_tune.loss.sds == 0.0:
            del self.guidance

        max_steps = int(total / bs) * epoch
        with tqdm(total=max_steps) as pbar:
            for e in range(epoch):
                for i in range(0, total, bs):
                    self.step = e * int(total / bs) + int(i / bs)
                    # self.update(self.step)
                    batch = get_dict_slice(all_data, i, i + bs)
                    out = self.renderer(batch)
                    if self.cfg.upsample_tune.loss.sds > 0.0:
                        guidance_out = self.guidance(
                            out["rgb"],
                            self.prompt_processor(),
                            rgb_as_latents=False,
                            elevation=batch["elevation"],
                            azimuth=batch["azimuth"],
                            camera_distance=batch["camera_distance"],
                        )
                    image_gt = upsampled_images[i : i + bs].to(self.cfg.device)

                    loss = 0.0
                    if self.cfg.upsample_tune.loss.sds > 0.0:
                        loss += (
                            self.cfg.upsample_tune.loss.sds * guidance_out["loss_sds"]
                        )
                    loss += self.cfg.upsample_tune.loss.rgb * self.image_loss_fn(
                        out["rgb"], image_gt
                    )

                    self.writer.add_image(
                        "train",
                        torch.cat([out["rgb"][0], image_gt[0]], dim=1),
                        self.step,
                        dataformats="HWC",
                    )
                    self.renderer.log(self.writer, self.step)

                    pbar.set_description(f"Upsample Tune|Iter: {self.step}/{max_steps}")
                    pbar.set_postfix(loss=f"{loss:.4f}")

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    del loss
                    del out
                    del image_gt
                    del batch

                gc.collect()
                torch.cuda.empty_cache()
            self.eval_video_step()
