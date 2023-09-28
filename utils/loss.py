import torch
from kornia.losses.ssim import SSIMLoss, ssim_loss
from torchmetrics import PearsonCorrCoef
import torch.nn as nn


def get_loss_fn(cfg):
    base_loss_fn = None
    if cfg.loss_fn == "l2":
        base_loss_fn = torch.nn.functional.mse_loss
    elif cfg.loss_fn == "l1":
        base_loss_fn = torch.nn.functional.l1_loss
    else:
        raise NotImplementedError

    def loss_fn(out, gt):
        loss = cfg.ssim_loss_mult * ssim_loss(
            out.moveaxis(-1, 0).unsqueeze(0),
            gt.moveaxis(-1, 0).unsqueeze(0),
            cfg.ssim_loss_win_size,
            reduction="mean",
        ) + (1 - cfg.ssim_loss_mult) * base_loss_fn(out, gt)

        return loss

    return loss_fn


def get_image_loss(ssim_weight=0.2, type="l1"):
    if type == "l1":
        base_loss_fn = torch.nn.functional.l1_loss
    elif type == "l2":
        base_loss_fn = torch.nn.functional.mse_loss
    else:
        raise NotImplementedError

    def loss_fn(out, gt):
        loss = ssim_weight * ssim_loss(
            out.moveaxis(-1, 1),
            gt.moveaxis(-1, 1),
            11,
            reduction="mean",
        ) + (1 - ssim_weight) * base_loss_fn(out, gt)

        return loss

    return loss_fn


def depth_loss(pearson, pred_depth, depth_gt, mask=None):
    if mask is None:
        mask = torch.ones_like(depth_gt[..., 0], dtype=torch.bool)
    bs = pred_depth.shape[0]
    # print("batch_size", bs)
    # print(mask.shape)
    loss = 0
    for pred_d, gt_d, m in zip(pred_depth, depth_gt, mask):
        # print(m.shape)
        # print(pred_d.shape)
        # print(gt_d.shape)
        # breakpoint()
        pred_d = torch.nan_to_num(pred_d)
        # m = m == 1
        co = pearson(pred_d[m].reshape(-1), gt_d[m].reshape(-1))
        loss += 1 - co
    return loss / bs


class LinearDepthLoss(nn.Module):
    def __init__(self):
        self.ab = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, pred, gt):
        return torch.nn.functional.l1_loss(pred * self.ab[0] + self.ab[1], gt)
