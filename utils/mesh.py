import trimesh
from pathlib import Path
import torch
import numpy as np
from vedo import Mesh


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.

    reference: https://github.com/mikedh/trimesh/issues/507#issuecomment-514973337
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def load_mesh_obj(obj_file, texture_file=None):
    mesh = Mesh(str(obj_file))
    if texture_file is not None:
        mesh.texture(texture_file)

    xyz = mesh.points()
    rgb = mesh.pointcolors.astype(np.float32) / 255.0

    return torch.from_numpy(xyz), torch.from_numpy(rgb)


def load_mesh_as_pcd(mesh_file, texture_file):
    mesh_file = Path(mesh_file)
    if mesh_file.suffix == ".obj":
        return load_mesh_obj(mesh_file, texture_file)
    else:
        raise NotImplementedError(f"Unknown mesh file {mesh_file}")


def load_mesh_as_pcd_trimesh(mesh_file, num_points):
    mesh = as_mesh(trimesh.load_mesh(mesh_file))
    n = num_points
    points = []
    while n > 0:
        p, _ = trimesh.sample.sample_surface_even(mesh, n)
        n -= p.shape[0]
        if n >= 0:
            points.append(p)
        else:
            points.append(p[:n])
    if len(points) > 1:
        points = np.concatenate(points, axis=0)
    else:
        points = points[0]
    points = torch.from_numpy(points.astype(np.float32))

    return points, torch.rand_like(points)
