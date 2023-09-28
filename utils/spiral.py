import random
import numpy as np
import torch
from data import CameraPoseProvider
from utils.camera import CameraInfo


def get_c2w_from_up_and_look_at(up, look_at, pos, return_pt=False):
    up = up / np.linalg.norm(up)
    z = look_at - pos
    z = z / np.linalg.norm(z)
    y = -up
    x = np.cross(y, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    c2w = np.zeros([3, 4], dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = pos

    if return_pt:
        c2w = torch.from_numpy(c2w)

    return c2w


def get_camera_path_fixed_elevation(
    n_frames, n_circles=1, camera_distance=2, elevation=45
):
    azimuth = np.linspace(0, 2 * np.pi * n_circles, n_frames)
    elevation_rad = np.deg2rad(elevation)

    x = camera_distance * np.cos(azimuth) * np.cos(elevation_rad)
    y = camera_distance * np.sin(azimuth) * np.cos(elevation_rad)
    z = camera_distance * np.sin(elevation_rad) * np.ones_like(x)

    up = np.array([0, 0, 1], dtype=np.float32)
    look_at = np.array([0, 0, 0], dtype=np.float32)
    pos = np.stack([x, y, z], axis=1)

    c2ws = []
    for i in range(n_frames):
        c2ws.append(
            get_c2w_from_up_and_look_at(
                up,
                look_at,
                pos[i],
            )
        )

    c2ws = np.stack(c2ws, axis=0)

    return c2ws


def get_camera_path_given_dataset(
    dataset: CameraPoseProvider, n_frames: int, n_circles: int
):
    azimuth = np.deg2rad(
        np.linspace(
            dataset.azimuth[0] * n_circles, dataset.azimuth[1] * n_circles, n_frames
        )
    )
    elevation_rad = np.deg2rad(np.mean(dataset.elevation))

    camera_distance = np.mean(dataset.camera_distance)

    x = camera_distance * np.cos(azimuth) * np.cos(elevation_rad)
    y = camera_distance * np.sin(azimuth) * np.cos(elevation_rad)
    z = camera_distance * np.sin(elevation_rad) * np.ones_like(x)

    up = np.array([0, 0, 1], dtype=np.float32)
    look_at = np.array([0, 0, 0], dtype=np.float32)
    pos = np.stack([x, y, z], axis=1)

    c2ws = []
    for i in range(n_frames):
        c2ws.append(
            get_c2w_from_up_and_look_at(
                up,
                look_at,
                pos[i],
            )
        )

    c2ws = np.stack(c2ws, axis=0)


def get_random_pose_fixed_elevation(camera_distance=2, elevation=45):
    azimuth_rad = random.random() * 2 * np.pi
    elevation_rad = np.deg2rad(elevation)

    pos = np.array(
        [
            camera_distance * np.cos(azimuth_rad) * np.cos(elevation_rad),
            camera_distance * np.sin(azimuth_rad) * np.cos(elevation_rad),
            camera_distance * np.sin(elevation_rad),
        ]
    )
    up = np.array([0, 0, 1], dtype=np.float32)
    look_at = np.array([0, 0, 0], dtype=np.float32)
    return get_c2w_from_up_and_look_at(up, look_at, pos)


def get_random_pose_given_dataset(dataset: CameraPoseProvider):
    azimuth_rad = np.deg2rad(np.random.uniform(*dataset.get_azimuth_bound))
    elevation_rad = np.deg2rad(np.random.uniform(*dataset.get_elevation_bound))
    camera_distance = np.mean(dataset.camera_distance)

    pos = np.array(
        [
            camera_distance * np.cos(azimuth_rad) * np.cos(elevation_rad),
            camera_distance * np.sin(azimuth_rad) * np.cos(elevation_rad),
            camera_distance * np.sin(elevation_rad),
        ]
    )
    up = np.array([0, 0, 1], dtype=np.float32)
    look_at = np.array([0, 0, 0], dtype=np.float32)
    return get_c2w_from_up_and_look_at(up, look_at, pos)


def fixed_elevation_spiral(renderer: torch.nn.Module):
    is_training = renderer.training
    renderer.eval()
    c2ws = get_camera_path_fixed_elevation(30, 1, 2.5, 45)
    c2ws = torch.from_numpy(c2ws).to(renderer.device)
    camera_info = CameraInfo.from_reso(256)
    outs = []
    use_bg = False
    with torch.no_grad():
        for c2w in c2ws:
            out = renderer.render_one(c2w, camera_info, use_bg)["rgb"]
            outs.append(out)

    outs = torch.stack(outs)

    if is_training:
        renderer.train()

    return (outs.cpu().numpy() * 255.0).astype(np.uint8)
