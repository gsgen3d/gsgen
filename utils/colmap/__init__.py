import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .read import read_points3D_binary, read_cameras_binary, read_images_binary
import time
import multiprocessing as mp
import torch
from utils.misc import tic, toc


def read_one_image(filename, conversion=True):
    filename = str(filename)
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if conversion:
        img = img.astype(np.float32) / 255.0

    return img


def read_images_mpl(filenames):
    num_workers = os.cpu_count()
    with mp.Pool(num_workers) as pool:
        images = pool.map(read_one_image, filenames)

    return np.array(images)


def read_pts_from_colmap(filename, verbose=True):
    if not isinstance(filename, str):
        filename = str(filename)
    obj = read_points3D_binary(filename)
    pts = []
    rgb = []
    for o in obj.values():
        pts.append(o.xyz)
        rgb.append(o.rgb)

    if verbose:
        print(f"Read {len(obj)} points from {filename}")

    rgb = np.array(rgb).astype(np.float32) / 255.0

    return np.array(pts), rgb


def colmap_to_open3d(filename):
    import open3d as o3d

    if not isinstance(filename, str):
        filename = str(filename)

    obj = read_points3D_binary(filename)

    xyz = []
    rgb = []

    for o in obj.values():
        xyz.append(o.xyz)
        rgb.append(o.rgb)

    xyz = np.array(xyz, dtype=np.float32)
    rgb = np.array(rgb, dtype=np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def stats(filename):
    if not isinstance(filename, str):
        filename = str(filename)
    obj = read_points3D_binary(filename)

    num_pts = len(obj)
    num_visible_img = []

    for o in obj.values():
        num_visible_img.append(len(o.image_ids))

    avg_num_visible_img = np.mean(num_visible_img)

    print(
        f"File: {filename}, #(points): {num_pts}, avg visible times: {avg_num_visible_img:.2f}"
    )


def read_cameras(filename):
    if not isinstance(filename, str):
        filename = str(filename)

    camera = read_cameras_binary(filename)[1]

    return camera


def read_images(filename, image_base_dir):
    if not isinstance(filename, str):
        filename = str(filename)
    images = read_images_binary(filename)
    image_base_dir = Path(image_base_dir)
    rot = []
    T = []
    imgs = []
    filenames = []
    for img in tqdm(images.values()):
        rot.append(img.qvec2rotmat())
        T.append(img.tvec)
        imgs.append(read_one_image(image_base_dir / img.name))
        # filenames.append(image_base_dir / img.name)

    # start = time.time()
    # imgs = read_images_mpl(filenames)
    # end = time.time()
    # print(f"Read {len(imgs)} images in {end - start:.2f} seconds")
    imgs = np.stack(imgs, axis=0)

    return np.array(rot), np.array(T), imgs


def read_images_v1(filename, image_base_dir):
    if not isinstance(filename, str):
        filename = str(filename)
    images = read_images_binary(filename)
    image_base_dir = Path(image_base_dir)
    rot = []
    T = []
    N = len(images)

    test_img = read_one_image(image_base_dir / list(images.values())[0].name)
    H, W = test_img.shape[:2]
    # imgs = np.empty([N, H, W, 3], dtype=np.uint8)
    imgs = np.empty([N, H, W, 3], dtype=np.uint8)

    filenames = []

    tic()
    for idx, img in tqdm(enumerate(images.values())):
        rot.append(img.qvec2rotmat())
        T.append(img.tvec)
        imgs[idx] = read_one_image(image_base_dir / img.name, conversion=False)
        filenames.append(img.name)
    toc("in [read image v1] load")

    tic()
    imgs = torch.from_numpy(imgs).to(torch.float32) / 255.0
    toc("type conversion")

    tic()
    rot = np.array(rot)
    T = np.array(T)
    toc("list to np.ndarray")

    return rot, T, imgs, filenames


def read_images_test(filename, image_base_dir):
    # return an empty np.array for image
    if not isinstance(filename, str):
        filename = str(filename)
    images = read_images_binary(filename)
    image_base_dir = Path(image_base_dir)
    rot = []
    T = []
    filenames = []
    for img in tqdm(images.values()):
        rot.append(img.qvec2rotmat())
        T.append(img.tvec)

    # start = time.time()
    # imgs = read_images_mpl(filenames)
    # end = time.time()

    return np.array(rot), np.array(T), None


def read_all(data_dir):
    data_dir = Path(data_dir)
    cam_bin = data_dir / "cameras.bin"
    img_bin = data_dir / "images.bin"
    pts_bin = data_dir / "points3D.bin"

    pos, rgb = read_pts_from_colmap(pts_bin)
    rot, T = read_images(img_bin)
    cam = read_cameras(cam_bin)
