import numpy as np
import torch
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import visdom


def draw_2d_circles(
    filename,
    mean,
    radius,
    depth,
    rgb,
    pixel_size_x,
    pixel_size_y,
    cx,
    cy,
    H,
    W,
    use_visdom=True,
):
    # mean: [N, 2]
    # radius: [N]
    N = mean.shape[0]
    x = mean[:, 0].cpu().numpy()
    y = mean[:, 1].cpu().numpy()
    r = radius.cpu().numpy()
    d = depth.squeeze().cpu().numpy()
    ids = np.argsort(d)
    x = ((x[ids] / pixel_size_x) + cx).astype(np.int32)
    y = ((y[ids] / pixel_size_y) + cy).astype(np.int32)
    r = (r / pixel_size_x).astype(np.int32)
    
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()

    rgb = (rgb[ids] * 255.0).astype(np.int32)

    img = np.zeros([H, W, 3], dtype=np.uint8)

    for idx in range(N):
        c = (int(rgb[idx][0]), int(rgb[idx][1]), int(rgb[idx][2]))
        cv2.circle(img, (x[idx], y[idx]), r[idx], c, -1)

    if use_visdom:
        vis = visdom.Visdom(env="vis_2ds")
        vis.image(
            img.transpose(2, 0, 1),
            win="2d gaussians",
            opts=dict(title="2d gaussians", caption=filename),
        )

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f"./tmp/{filename}.png", img)


def draw_heatmap_of_num_gaussians_per_tile(
    filename, tile_size, num_gaussians, n_tile_h, n_tile_w, H, W, use_visdom=True
):
    freq = np.zeros([H, W, 1], dtype=np.int32)

    for y in range(n_tile_h):
        for x in range(n_tile_w):
            tile_id = y * n_tile_w + x
            n_gaussians = num_gaussians[tile_id].item()
            y_min, y_max = y * tile_size, min((y + 1) * tile_size, H)
            x_min, x_max = x * tile_size, min((x + 1) * tile_size, W)
            freq[y_min:y_max, x_min:x_max] = n_gaussians

    freq = freq.astype(np.float32)
    freq = freq / freq.max()
    freq = (freq * 255.0).astype(np.uint8)

    img = cv2.applyColorMap(freq, cv2.COLORMAP_VIRIDIS)

    if use_visdom:
        vis = visdom.Visdom(env="vis_2ds")
        vis.image(
            img.transpose(2, 0, 1),
            win="n_gaussians_per_tile",
            opts=dict(title="n_gaussians_per_tile", caption=filename),
        )

    cv2.imwrite(f"./tmp/{filename}.png", img)

    return img


def draw_heatmap_of_num_gaussians_per_tile_with_offset(
    filename, tile_size, offset, n_tile_h, n_tile_w, H, W, use_visdom=True
):
    freq = np.zeros([H, W, 1], dtype=np.int32)

    for y in range(n_tile_h):
        for x in range(n_tile_w):
            tile_id = y * n_tile_w + x
            n_gaussians = offset[tile_id + 1].item() - offset[tile_id].item()
            y_min, y_max = y * tile_size, min((y + 1) * tile_size, H)
            x_min, x_max = x * tile_size, min((x + 1) * tile_size, W)
            freq[y_min:y_max, x_min:x_max] = n_gaussians

    freq = freq.astype(np.float32)
    freq = freq / freq.max()
    freq = (freq * 255.0).astype(np.uint8)

    img = cv2.applyColorMap(freq, cv2.COLORMAP_VIRIDIS)

    if use_visdom:
        vis = visdom.Visdom(env="vis_2ds")
        vis.image(
            img.transpose(2, 0, 1),
            win="n_gaussians_per_tile_with_offset",
            opts=dict(title="n_gaussians_per_tile_with_offset", caption=filename),
        )

    cv2.imwrite(f"./tmp/{filename}.png", img)
