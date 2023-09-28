import numpy as np
import torch
import faiss

def cov_init(pts, k=3):
    ## set cov to mean distance of nearest k points
    if not isinstance(pts, torch.Tensor):
        pts = torch.from_numpy(pts).to("cuda")

    # pts = pts.to("cuda")
    # dist = torch.cdist(pts, pts)
    # topk = torch.topk(dist, k=k, dim=1, largest=False)

    # return topk.mean(axis=)
    res = faiss.StandardGpuResources()
    # pts = pts.to("cuda")
    index = faiss.IndexFlatL2(pts.shape[1])
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index_flat.add(pts.cpu())
    D, _ = gpu_index_flat.search(pts, k + 1)

    return torch.from_numpy(D[..., 1:].mean(axis=1))


@torch.no_grad()
def alpha_center_anealing_init(mean, center=None, alpha_max=1.0):
    if center is None:
        center = torch.zeros_like(mean.data[0])

    pos = mean - center[None, ...]
    max_dist = pos.norm(dim=-1).max() + 1e-5

    std = torch.std(pos).item()

    alpha = alpha_max * torch.exp(-torch.norm(pos, dim=-1) ** 2 / std)

    return alpha


@torch.no_grad()
def alpha_trunc_init(mean, alpha_inside, alpha_outside, radius, center=None):
    # intialize alpha to be alpha_inside inside the shape and alpha_outside outside
    alpha = torch.ones_like(mean[..., 0])

    if center is None:
        center = torch.zeros_like(mean.data[0])

    mask = (mean - center[None, ...]).norm(dim=-1) < radius

    alpha[mask] = alpha_inside
    alpha[~mask] = alpha_outside

    return alpha
