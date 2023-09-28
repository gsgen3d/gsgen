import numpy as np
import torch
from utils.typing import *
import torch.nn.functional as F
from kornia.geometry.depth import depth_to_3d

pytorch3d_capable = True
try:
    import pytorch3d
    from pytorch3d.ops import estimate_pointcloud_normals
    from pytorch3d.ops import sample_farthest_points
    from pytorch3d.ops import knn_points
except ImportError:
    pytorch3d_capable = False


def shifted_expotional_decay(a, b, c, r):
    return a * torch.exp(-b * r) + c


def perpendicular_component(x: Float[Tensor, "B C H W"], y: Float[Tensor, "B C H W"]):
    # get the component of x that is perpendicular to y
    eps = torch.ones_like(x[:, 0, 0, 0]) * 1e-6
    return (
        x
        - (
            torch.mul(x, y).sum(dim=[1, 2, 3])
            / torch.maximum(torch.mul(y, y).sum(dim=[1, 2, 3]), eps)
        ).view(-1, 1, 1, 1)
        * y
    )


ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]


def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()


def angle_bisector(a, b):
    return F.normalize(F.normalize(a, dim=-1) + F.normalize(b, dim=-1), dim=-1)


def estimate_normal(
    pos, neighborhood_size: int = 50, disambiguate_directions: bool = True
):
    if not pytorch3d_capable:
        raise ImportError(
            "pytorch3d is not installed, which is required for normal estimation"
        )

    return estimate_pointcloud_normals(
        pos[None, ...], neighborhood_size, disambiguate_directions
    )[0]


@torch.no_grad()
def farthest_point_sampling(mean: torch.Tensor, K, random_start_point=False):
    if not pytorch3d_capable:
        raise ImportError("pytorch3d is not installed, which is required for FPS")

    if mean.ndim == 2:
        L = torch.tensor(mean.shape[0], dtype=torch.long).to(mean.device)
        pts, indices = sample_farthest_points(
            mean[None, ...],
            L[None, ...],
            K,
            random_start_point=random_start_point,
        )
        return pts[0], indices[0]
    elif mean.ndim == 3:
        # mean: [B, L, 3]
        B = mean.shape[0]
        L = torch.tensor(mean.shape[1], dtype=torch.long).to(mean.device)
        pts, indices = sample_farthest_points(
            mean,
            L[None, ...].repeat(B),
            K,
            random_start_point=random_start_point,
        )

        return pts, indices


@torch.no_grad()
def nearest_neighbor(mean: torch.Tensor):
    if not pytorch3d_capable:
        raise ImportError(
            "pytorch3d is not installed, which is required for nearest neighbor"
        )

    _, idx, nn = knn_points(mean[None, ...], mean[None, ...], K=2, return_nn=True)
    # nn: Tensor of shape (N, P1, K, D)

    # take the index 1 since index 0 is the point itself
    return nn[0, :, 1, :], idx[..., 1][0]


@torch.no_grad()
def K_nearest_neighbors(
    mean: torch.Tensor, K: int, query: Optional[torch.Tensor] = None
):
    if not pytorch3d_capable:
        raise ImportError("pytorch3d is not installed, which is required for KNN")
    # TODO: finish this
    if query is None:
        query = mean
    _, idx, nn = knn_points(query[None, ...], mean[None, ...], K=K, return_nn=True)

    return nn[0, :, 1:, :], idx[0, :, 1:]


def distance_to_gaussian_surface(mean, svec, rotmat, query):
    xyz = query - mean
    # TODO: check here
    # breakpoint()
    xyz = torch.einsum("bij,bj->bi", rotmat.transpose(-1, -2), xyz)
    xyz = F.normalize(xyz, dim=-1)
    z = xyz[..., 2]
    y = xyz[..., 1]
    x = xyz[..., 0]
    r_xy = torch.sqrt(x**2 + y**2 + 1e-10)
    cos_theta = z
    sin_theta = r_xy
    cos_phi = x / r_xy
    sin_phi = y / r_xy

    d2 = svec[..., 0] ** 2 * cos_phi**2 + svec[..., 1] ** 2 * sin_phi**2
    r2 = svec[..., 2] ** 2 * cos_theta**2 + d2**2 * sin_theta**2

    return torch.sqrt(r2 + 1e-10)


def linear_correlation(x, y):
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    x = x / torch.norm(x, dim=-1, keepdim=True)
    y = y / torch.norm(y, dim=-1, keepdim=True)
    return (x * y).sum(dim=-1)


@torch.no_grad()
def lift_to_3d(depth, camera_info, c2w=None):
    if depth.ndim == 3:
        depth = depth[None, ...]
    if depth.shape[-1] == 1:
        depth = depth.moveaxis(-1, 1)
    camera_mtx = camera_info.get_camera_intrinsic()[None, ...]

    xyz = depth_to_3d(depth, camera_mtx, normalize_points=False)[0]  # [3, H, W]
    xyz = xyz.permute(1, 2, 0)  # [H, W, 3]

    if c2w is not None:
        if isinstance(c2w, np.ndarray):
            c2w = torch.from_numpy(c2w).to(xyz)
        c2w = c2w.to(xyz)
        xyz = torch.einsum("ij,hwj->hwi", c2w[:3, :3], xyz) + c2w[:3, 3]

    # print(xyz)

    return xyz


def compute_shaded_color(
    light_pos, light_color, surface_normal, surface_color, mean, cam_pos
):
    ab = angle_bisector(light_pos - mean, cam_pos - mean)
    # backface culling
    dot = (ab * surface_normal).sum(dim=-1).abs().clamp(min=0.0, max=1.0)

    return light_color * dot[..., None] * surface_color
