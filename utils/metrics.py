from typing import Any
import numpy as np
import torch
import kornia
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Metrics:
    def __init__(self, device="cuda") -> None:
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity().to(device)

    def evaluate(self, pred, gt):
        pred = pred.clamp(min=0.0, max=1.0).moveaxis(-1, 0).unsqueeze(0)
        gt = gt.clamp(min=0.0, max=1.0).moveaxis(-1, 0).unsqueeze(0)
        psnr = self.psnr(pred, gt)
        ssim = self.ssim(
            pred,
            gt,
            data_range=1.0,
        )
        ssim2 = self.ssim(
            pred,
            gt,
            data_range=2.0,
        )
        lpips = self.lpips(pred, gt)

        metrics = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "ssim2": ssim2.item(),
            "lpips": lpips.item(),
        }

        return metrics

    def __call__(self, pred, gt):
        return self.evaluate(pred, gt)
