import torch
import numpy as np
from utils.camera import CameraInfo
from torch.utils.data import Dataset
from data import CameraPoseProvider, get_c2w_from_up_and_look_at


class SingleViewCameraPoseProvider(CameraPoseProvider):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.original_view_prob = self.cfg.original_view_prob

        self.original_elevation = self.cfg.get("original_elevation", 0.0)
        self.original_azimuth = self.cfg.get("original_azimuth", 0.0)
        self.original_camera_distance = self.cfg.get("original_camera_distance", 2.0)
        self.original_focal = self.cfg.get("original_focal", 1.0)
        look_at = np.array([0.0, 0.0, 0.0])
        pos = np.array([self.original_camera_distance, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        c2w = torch.from_numpy(get_c2w_from_up_and_look_at(up, look_at, pos)).to(
            torch.float32
        )

        self.original_out = {
            "c2w": c2w,
            "camera_info": CameraInfo.from_reso(self.get_reso),
            "elevation": 0.0,
            "azimuth": 0.0,
            "camera_distance": self.original_camera_distance,
            "is_original_view": True,
            "light_pos": np.zeros(3),
            "light_color": torch.ones(3),
        }

    def sample_one(self):
        if np.random.random() < self.original_view_prob:
            return self.original_out
        else:
            out = super().sample_one()
            out.update({"is_original_view": False})
            return out
