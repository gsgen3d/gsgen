from pathlib import Path
import numpy as np
import torch
from utils.misc import print_info
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor
from utils.ops import lift_to_3d, farthest_point_sampling
from utils.camera import CameraInfo
from utils.mesh import load_mesh_as_pcd, load_mesh_as_pcd_trimesh
from rich.console import Console

console = Console()


def nearest_neighbor_initialize(pts, k=3):
    import faiss

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


def get_qvec(cfg):
    qvec = torch.zeros(cfg.num_points, 4, dtype=torch.float32)
    qvec[:, 0] = 1.0
    return qvec


def get_svec(cfg):
    svec = torch.ones(cfg.num_points, 3, dtype=torch.float32) * cfg.svec_val
    return svec


def get_alpha(cfg):
    alpha = torch.ones(cfg.num_points, dtype=torch.float32) * cfg.alpha_val
    return alpha


def base_initialize(cfg):
    initial_values = {}
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["svec"] = get_svec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    initial_values["color"] = torch.rand(cfg.num_points, 3, dtype=torch.float32)
    initial_values["mean"] = (
        torch.randn(cfg.num_points, 3, dtype=torch.float32) * cfg.mean_std
    )

    return initial_values


def unisphere_initialize(cfg):
    R = cfg.mean_std
    N = cfg.num_points
    theta = torch.rand(N) * 2 * np.pi
    phi = torch.rand(N)
    phi = torch.acos(1 - 2 * phi)
    x = R * torch.sin(phi) * torch.cos(theta)
    y = R * torch.sin(phi) * torch.sin(theta)
    z = R * torch.cos(phi)

    initial_values = {}
    initial_values["mean"] = torch.stack([x, y, z], dim=1)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["svec"] = get_svec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    initial_values["color"] = torch.rand(cfg.num_points, 3, dtype=torch.float32)

    return initial_values


def semisphere_initialize(cfg):
    R = cfg.mean_std
    N = cfg.num_points
    theta = torch.rand(N) * np.pi + np.pi / 2.0
    phi = torch.rand(N)
    phi = torch.acos(1 - 2 * phi)
    x = R * torch.sin(phi) * torch.cos(theta)
    y = R * torch.sin(phi) * torch.sin(theta)
    z = R * torch.cos(phi)

    initial_values = {}
    initial_values["mean"] = torch.stack([x, y, z], dim=1)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["svec"] = get_svec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    initial_values["color"] = torch.rand(cfg.num_points, 3, dtype=torch.float32)

    return initial_values


def point_e_intialize(cfg):
    from utils.point_e_helper import point_e_generate_pcd_from_text

    prompt = cfg.prompt
    pcd = point_e_generate_pcd_from_text(prompt, 4096)
    xyz, rgb = pcd[:, :3], pcd[:, 3:]

    if cfg.num_points > 4096:
        if cfg.get("random_exceed", False):
            indices = torch.randint(
                0, xyz.size(0), (cfg.num_points,), device=xyz.device
            )
            xyz = xyz[indices]
            rgb = rgb[indices]
        else:
            extra_xyz = (
                torch.randn(
                    cfg.num_points - 4096, 3, dtype=torch.float32, device=xyz.device
                )
                * cfg.mean_std
            )
            extra_rgb = torch.rand(
                cfg.num_points - 4096, 3, dtype=torch.float32, device=rgb.device
            )
            xyz = torch.cat([xyz, extra_xyz], dim=0)
            rgb = torch.cat([rgb, extra_rgb], dim=0)

    xyz -= xyz.mean(dim=0, keepdim=True)

    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-5)
    xyz = xyz * cfg.mean_std

    if cfg.get("facex", False):
        # align the point cloud to the x axis
        console.print("[red]will align the point cloud to the x axis")
        x, y, z = xyz.chunk(3, dim=-1)
        xyz = torch.cat([-y, x, z], dim=-1)

    if cfg.get("random_color", False):
        console.print("[red]will use random color")
        rgb = torch.rand_like(rgb)

    if cfg.get("white_color", False):
        console.print("[red]will make all the gaussians white, for experimental usage")
        rgb = torch.ones_like(rgb) * 0.7

    z_scale = cfg.get("z_scale", 1.0)
    xyz[..., 2] *= z_scale

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    # breakpoint()
    initial_values["svec"] = get_svec(cfg)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    return initial_values


# moved to utils/debug.py
# def debug_initialize(debug_flag):
#     initial_values = {}
#     if debug_flag == "one":
#         # A big gaussian in the center
#         initial_values["mean"] = torch.tensor([[0.0, 0.0, 0.0]])
#         initial_values["svec"] = torch.tensor([[0.1, 0.1, 0.3]])
#         initial_values["qvec"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
#         initial_values["color"] = torch.tensor([[0.01, 0.01, 0.99]])
#         initial_values["alpha"] = torch.tensor([0.8])
#     else:
#         raise NotImplementedError

#     return initial_values


def point_cloud_initialize(cfg):
    initial_values = {}
    pcd = Path(cfg.pcd)
    assert pcd.exists(), f"point cloud file {pcd} does not exist"
    extension_name = pcd.suffix
    if extension_name == ".npy":
        pcd = torch.from_numpy(np.load(pcd))
    elif extension_name in [".pt", ".pth"]:
        pcd = torch.load(pcd)
    else:
        raise ValueError(f"Unknown point cloud file extension {extension_name}")

    xyz = pcd[:, :3]
    rgb = pcd[:, 3:]
    cfg.num_points = xyz.shape[0]
    num_points = xyz.shape[0]
    if cfg.svec_val > 0.0:
        svec = torch.ones(num_points, 3, dtype=torch.float32) * cfg.svec_val
    else:
        svec = nearest_neighbor_initialize(xyz, k=3)[..., None].repeat(1, 3)
    alpha = get_alpha(cfg)
    qvec = get_qvec(cfg)

    # xyz[..., 0], xyz[..., 1] = xyz[..., 1], xyz[..., 0]
    x, y, z = xyz.chunk(3, dim=-1)
    xyz = torch.cat([-y, x, z], dim=-1)

    # breakpoint()
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    initial_values["svec"] = svec
    initial_values["qvec"] = qvec
    initial_values["alpha"] = alpha
    initial_values["raw"] = False

    return initial_values


def mesh_initlization(cfg):
    mesh_path = Path(cfg.mesh)
    assert mesh_path.exists(), f"Mesh path {mesh_path} does not exist"
    xyz, rgb = load_mesh_as_pcd_trimesh(mesh_path, cfg.num_points)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    xyz -= xyz.mean(dim=0, keepdim=True)

    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-5)
    xyz = xyz * cfg.mean_std

    # if xyz.shape[0] > cfg.num_points:
    #     _, idx = farthest_point_sampling(xyz, cfg.num_points)
    #     xyz = xyz[idx]
    #     rgb = rgb[idx]
    # else:
    #     cfg.num_points = xyz.shape[0]

    if cfg.get("flip_yz", False):
        console.print("[red]will flip the y and z axis")
        x, y, z = xyz.chunk(3, dim=-1)
        xyz = torch.cat([x, z, y], dim=-1)

    if cfg.get("flip_xy", False):
        console.print("[red]will flip the x and y axis")
        x, y, z = xyz.chunk(3, dim=-1)
        xyz = torch.cat([y, x, z], dim=-1)

    if cfg.svec_val > 0.0:
        svec = get_svec(cfg)
    else:
        svec = nearest_neighbor_initialize(xyz, k=3)[..., None].repeat(1, 3)
    alpha = get_alpha(cfg)
    qvec = get_qvec(cfg)

    if cfg.get("random_color", True):
        rgb = torch.rand_like(rgb)

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    initial_values["svec"] = svec
    initial_values["qvec"] = qvec
    initial_values["alpha"] = alpha
    initial_values["raw"] = False

    return initial_values


def from_ckpt(cfg):
    ckpt_path = Path(cfg.ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if cfg is None:
        cfg = {}
    if not "params" in ckpt:
        # case for loading only renderer ckpt
        new_cfg = OmegaConf.create(ckpt["cfg"])
        new_cfg.update(cfg)
        del ckpt["cfg"]
        cfg = new_cfg
    else:
        new_cfg = OmegaConf.create(ckpt["cfg"]).renderer
        new_cfg.update(cfg)
        ckpt = ckpt["params"]
    # This following two lines cause a bug when loading from ckpt, so I commented them out
    # if ckpt["color"].max() > 1 or ckpt["color"].min() < 0:
    #     ckpt["color"] = torch.sigmoid(ckpt["color"])
    ckpt["raw"] = True

    return ckpt


def image_initialize(cfg, **kwargs):
    # will generate 2 * cfg.num_points gaussian, half for front view, others will be optmized for front view
    num_points = cfg.num_points
    image = kwargs["image"].squeeze()
    # TODO: finish this
    depth_map = kwargs["depth_map"]
    c2w = kwargs["c2w"]
    camera_info = kwargs["camera_info"]
    mask = kwargs["mask"].squeeze()

    camera_info = CameraInfo.from_reso(depth_map.shape[1])
    pcd = lift_to_3d(depth_map, camera_info, c2w)
    pcd = pcd[mask]
    rgb = image[mask].to(pcd.device)
    print(pcd[..., 0].max())
    print(pcd[..., 0].min())
    print(pcd[..., 0].std())

    # breakpoint()
    if pcd.shape[0] > num_points:
        _, idx = farthest_point_sampling(pcd, num_points)
        # idx = idx.to(pcd.device)
        pcd = pcd[idx]
        rgb = rgb[idx]

    additional_pts = semisphere_initialize(cfg)

    cfg.num_points = pcd.shape[0]

    image_base_pts = {
        "mean": pcd,
        "color": rgb,
        "svec": get_svec(cfg),
        "qvec": get_qvec(cfg),
        "alpha": get_alpha(cfg),
    }

    initialize_values = {}
    for key in image_base_pts:
        initialize_values[key] = torch.cat(
            [image_base_pts[key], additional_pts[key]], dim=0
        )

    if cfg.get("grad_mask", False):
        grad_mask = torch.ones_like(initialize_values["mean"][..., 0])
        grad_mask[: pcd.shape[0]] = 0.0
        initialize_values["mask"] = grad_mask

    return initialize_values


def point_e_image_initialize(cfg, **kwargs):
    from utils.point_e_helper import point_e_generate_pcd_from_image

    if "image" in kwargs:
        image = kwargs["image"].squeeze()
    else:
        assert hasattr(cfg, "image"), "image not found in cfg"
        image = str(cfg.image)
    pcd = point_e_generate_pcd_from_image(
        image, cfg.num_points, cfg.get("base_name", None)
    )
    xyz, rgb = pcd[:, :3], pcd[:, 3:]
    xyz = xyz / (xyz.norm(dim=-1).max() + 1e-5)
    xyz = xyz * cfg.mean_std

    if cfg.get("facex", False):
        # align the point cloud to the x axis
        x, y, z = xyz.chunk(3, dim=-1)
        xyz = torch.cat([-y, x, z], dim=-1)

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["color"] = rgb
    # breakpoint()
    initial_values["svec"] = get_svec(cfg)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    return initial_values


def unbounded_initialize(cfg):
    R = cfg.mean_std
    N = cfg.num_points
    theta = torch.rand(N) * 2 * np.pi
    phi = torch.rand(N)
    phi = torch.acos(1 - 2 * phi)
    x = R * torch.sin(phi) * torch.cos(theta)
    y = R * torch.sin(phi) * torch.sin(theta)
    z = R * torch.cos(phi)

    initial_values = {}
    initial_values["mean"] = torch.stack([x, y, z], dim=1)
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["svec"] = get_svec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    initial_values["color"] = torch.rand(cfg.num_points, 3, dtype=torch.float32)

    return initial_values


def box_initialize(cfg):
    L = cfg.mean_std
    N = cfg.num_points
    u = (torch.rand(N) * 2 - 1) * L
    v = (torch.rand(N) * 2 - 1) * L
    w = torch.ones_like(u) * L / 2
    w[::2] = -w[::2]
    xyz = torch.stack([u, v, w], dim=1)
    for i in range(N):
        permutations = torch.randperm(3)
        xyz[i] = xyz[i][permutations]

    initial_values = {}
    initial_values["mean"] = xyz
    initial_values["qvec"] = get_qvec(cfg)
    initial_values["svec"] = get_svec(cfg)
    initial_values["alpha"] = get_alpha(cfg)

    initial_values["color"] = torch.rand(cfg.num_points, 3, dtype=torch.float32)

    return initial_values


def initialize(cfg, **kwargs):
    init_type = cfg.type
    if init_type == "base":
        return base_initialize(cfg)
    elif init_type == "unisphere":
        return unisphere_initialize(cfg)
    elif init_type == "point_e":
        return point_e_intialize(cfg)
    elif init_type == "ckpt":
        return from_ckpt(cfg)
    elif init_type == "image":
        return image_initialize(cfg, **kwargs)
    elif init_type == "point_cloud":
        return point_cloud_initialize(cfg, **kwargs)
    elif init_type == "mesh":
        return mesh_initlization(cfg, **kwargs)
    elif init_type == "point_e_image":
        return point_e_image_initialize(cfg, **kwargs)
    elif init_type == "unbounded":
        return unbounded_initialize(cfg)
    elif init_type == "box":
        return box_initialize(cfg)
    else:
        raise NotImplementedError(f"Unknown initialization type: {init_type}")
