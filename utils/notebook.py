import trimesh
from trimesh.viewer import scene_to_notebook
import numpy as np
import torch


def show_pointcloud_in_notebook(xyz, rgb):
    if not isinstance(xyz, np.ndarray):
        xyz = xyz.cpu().numpy()
    if not isinstance(rgb, np.ndarray):
        rgb = rgb.cpu().numpy()
    
    pcd = trimesh.PointCloud(vertices=xyz, colors=rgb)
    scene = trimesh.Scene([pcd])
    scene_to_notebook(scene)
