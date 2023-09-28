import bisect
import torch
import torch.nn.functional as F
import numpy as np
from utils.camera import CameraInfo
from torch.utils.data import Dataset
from utils.misc import to_primitive
from einops import repeat
from rich.console import Console

console = Console()


def get_c2w_from_up_and_look_at(up, look_at, pos):
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

    return c2w


class CameraPoseProvider(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.center = np.array(self.cfg.center)
        self.center = np.array(self.center)
        self.center_aug_std = self.cfg.center_aug_std

        self.azimuth = to_primitive(self.cfg.azimuth)
        self.elevation = self.cfg.elevation
        self.azimuth_warmup = self.cfg.azimuth_warmup
        self.elevation_warmup = self.cfg.elevation_warmup
        self.camera_distance = self.cfg.camera_distance
        self.reso = self.cfg.reso
        self.reso_milestones = to_primitive(self.cfg.reso_milestones)
        self.reso_milestones = [-1] + self.reso_milestones

        if self.cfg.get("focal_milestones", None) is None:
            self.focal_milestones = [-1]
        else:
            self.focal_milestones = to_primitive(self.cfg.focal_milestones)
            self.focal_milestones = [-1] + self.focal_milestones
        self.focal = to_primitive(self.cfg.focal)
        if not isinstance(self.focal[0], list):
            self.focal = [self.focal]
        assert len(self.reso_milestones) == len(self.reso)
        assert len(self.focal_milestones) == len(self.focal)

        self.up = np.array([0.0, 0.0, 1.0])

        self.near_plane = self.cfg.near_plane
        self.far_plane = self.cfg.far_plane

        self.step = 0
        self.max_steps = self.cfg.max_steps

        self.stratified_on_azimuth = self.cfg.get("stratified_on_azimuth", False)
        if self.stratified_on_azimuth:
            self.bs = self.cfg.get("batch_size", 1)
            self.bin_idx = 0

        self.light_sample = self.cfg.get("light_sample", "dreamfusion")
        self.light_distance_range = self.cfg.get("light_distance_range", [2.5, 3.5])
        self.light_aug_std = self.cfg.get("light_aug_std", 0.3)

    def update(self, step):
        # TODO: add warm_up and reso milestone
        self.step = step

    @property
    def get_reso(self):
        index = bisect.bisect(self.reso_milestones, self.step) - 1
        return self.reso[index]

    @property
    def get_azimuth_bound(self):
        if not self.stratified_on_azimuth:
            return [
                self.azimuth[0]
                * min(self.step / (self.azimuth_warmup * self.max_steps + 1e-5), 1.0),
                self.azimuth[1]
                * min(self.step / (self.azimuth_warmup * self.max_steps + 1e-5), 1.0),
            ]
        else:
            self.bin_idx = (self.bin_idx + 1) % self.bs
            self.bins = np.linspace(
                self.azimuth[0]
                * min(self.step / (self.azimuth_warmup * self.max_steps + 1e-5), 1.0),
                self.azimuth[1]
                * min(self.step / (self.azimuth_warmup * self.max_steps + 1e-5), 1.0),
                self.bs + 1,
            )
            return [self.bins[self.bin_idx], self.bins[self.bin_idx + 1]]

    @property
    def get_elevation_bound(self):
        return [
            self.elevation[0]
            * min(self.step / (self.elevation_warmup * self.max_steps + 1e-5), 1.0),
            self.elevation[1]
            * min(self.step / (self.elevation_warmup * self.max_steps + 1e-5), 1.0),
        ]

    @property
    def get_focal_bound(self):
        index = bisect.bisect(self.focal_milestones, self.step) - 1
        return self.focal[index]

    def __getitem__(self, index):
        return self.sample_one()

    def __len__(self):
        # return self.cfg.max_steps * self.cfg.batch_size + 100
        return torch.iinfo(torch.long).max

    def collate(self, batch):
        out = {}
        for key in batch[0].keys():
            if not isinstance(batch[0][key], CameraInfo):
                out[key] = torch.utils.data.default_collate(
                    [item[key] for item in batch]
                )
            else:
                out[key] = [item[key] for item in batch]
        return out

    def get_default_camera_info(self):
        return CameraInfo(
            np.mean(self.get_focal_bound) * self.reso[-1],
            np.mean(self.get_focal_bound) * self.reso[-1],
            self.reso[-1] / 2.0,
            self.reso[-1] / 2.0,
            self.reso[-1],
            self.reso[-1],
            self.near_plane,
            self.far_plane,
        )

    def sample_one(self):
        reso = self.get_reso
        camera_distance = np.random.uniform(*self.camera_distance)

        if self.cfg.elevation_real_uniform:
            elevation = self.get_elevation_bound
            elevation_range_percent = [
                (elevation[0] + 90.0) / 180.0,
                (elevation[1] + 90.0) / 180.0,
            ]
            elevation_rad = np.arcsin(
                2
                * (
                    np.random.rand()
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation = np.rad2deg(elevation_rad)
        else:
            elevation = np.random.uniform(*self.get_elevation_bound)
            elevation_rad = np.deg2rad(elevation)

        azimuth = np.random.uniform(*self.get_azimuth_bound)
        azimuth_rad = np.deg2rad(azimuth)

        x = camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = camera_distance * np.sin(elevation_rad)

        center = self.center + np.random.randn(3) * self.center_aug_std

        pos = np.array([x, y, z])

        c2w = torch.from_numpy(
            get_c2w_from_up_and_look_at(
                self.up,
                center,
                pos,
            )
        ).to(torch.float32)

        focal = np.random.uniform(*self.get_focal_bound) * reso

        camera_info = CameraInfo(
            focal,
            focal,
            reso / 2.0,
            reso / 2.0,
            reso,
            reso,
            self.near_plane,
            self.far_plane,
        )

        # sample light position
        light_distances = np.random.uniform(*self.light_distance_range)
        if self.light_sample == "dreamfusion":
            light_direction = pos + np.random.randn(3) * self.light_aug_std
            light_direction /= np.linalg.norm(light_direction)
            # get light position by scaling light direction by light distance
            light_positions = (light_direction * light_distances).astype(np.float32)
        elif self.light_sample == "magic3d":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown light sample method {self.light_sample}")

        out = {
            # "pos": pos,
            "c2w": c2w,
            "camera_info": camera_info,
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distance": camera_distance,
            "light_pos": light_positions,
            "light_color": torch.ones(3),
        }

        return out

    def sample_with_clip(self):
        pass

    def get_batch(self, bacth_size):
        # call sample_one bacth_size times
        batch = []
        for _ in range(bacth_size):
            batch.append(self.sample_one())
        return self.collate(batch)

    def get_uniform_batch(self, batch_size):
        # get batch_size samples from the whole dataset, samples uniformly in azimuth
        # TODO: finish this
        reso = self.get_reso
        camera_distance = np.random.uniform(*self.camera_distance, size=batch_size)

        if self.cfg.elevation_real_uniform:
            elevation = self.get_elevation_bound
            elevation_range_percent = [
                (elevation[0] + 90.0) / 180.0,
                (elevation[1] + 90.0) / 180.0,
            ]
            elevation_rad = np.arcsin(
                2
                * (
                    np.random.rand(batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation = np.rad2deg(elevation_rad)
        else:
            elevation = np.random.uniform(*self.get_elevation_bound, size=batch_size)
            elevation_rad = np.deg2rad(elevation)

        azimuth = np.linspace(*self.get_azimuth_bound, batch_size)
        azimuth_rad = np.deg2rad(azimuth)

        x = camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = camera_distance * np.sin(elevation_rad)

        center = repeat(self.center, "d -> b d", b=batch_size)
        center = center + np.random.randn(batch_size, 3) * self.center_aug_std

        c2ws = []
        camera_infos = []
        for xx, yy, zz, cc in zip(x, y, z, center):
            pos = np.array([xx, yy, zz])
            c2ws.append(torch.from_numpy(get_c2w_from_up_and_look_at(self.up, cc, pos)))
            camera_infos.append(CameraInfo.from_reso(reso))

        c2w = torch.stack(c2ws)

        out = {
            "c2w": c2w,
            "camera_info": camera_infos,
            "elevation": torch.from_numpy(elevation),
            "azimuth": torch.from_numpy(azimuth),
            "camera_distance": torch.from_numpy(camera_distance),
        }

        return out

    def set_reso(self, reso: int):
        self.reso = [reso]

    def log(self, writer, step):
        writer.add_scalar("data/azimuth_min", self.get_azimuth_bound[0], step)
        writer.add_scalar("data/azimuth_max", self.get_azimuth_bound[1], step)
        writer.add_scalar("data/elevation_min", self.get_elevation_bound[0], step)
        writer.add_scalar("data/elevation_max", self.get_elevation_bound[1], step)
        writer.add_scalar("data/reso", self.get_reso, step)
        writer.add_scalar("data/focal_min", self.get_focal_bound[0], step)
        writer.add_scalar("data/focal_max", self.get_focal_bound[1], step)


from .sit3d import SingleViewCameraPoseProvider
