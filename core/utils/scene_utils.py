import json
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import numpy.typing as npt
import torch
from plyfile import PlyData, PlyElement

from core.utils.camera_utils import fov2focal, get_world2view
from core.utils.sh_utils import RGB2SH, SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: torch.FloatTensor
    T: torch.FloatTensor
    fovX: float
    fovY: float
    image: torch.ByteTensor  # HWC, uint8
    image_path: Path
    image_name: str
    width: int
    height: int
    masks: torch.Tensor | None
    depth: torch.Tensor | None = None
    cx: float | None = None
    cy: float | None = None
    normal: torch.Tensor | None = None


class BasicPointCloud(NamedTuple):
    points: torch.Tensor
    colors: torch.Tensor
    normals: torch.Tensor
    instance_ids: torch.Tensor | None = None  # SAM majority-voted instance label per point


class NerfNormalization(NamedTuple):
    translate: torch.Tensor  # tensor.Size([3])
    radius: float


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud | None
    train_cameras: list[CameraInfo]
    test_cameras: list[CameraInfo]
    nerf_normalization: NerfNormalization
    ply_path: Path
    is_pcd_random: bool


def search_for_max_iteration(folder: Path):
    saved_iters = [
        int(fname.stem.rsplit("_", 1)[-1])
        for fname in folder.iterdir()
        if fname.is_dir()
    ]
    return max(saved_iters)


def get_center_and_diag(cam_centers: list[torch.FloatTensor]):
    centers = torch.stack(cam_centers, dim=1)  # type: ignore
    avg_cam_center = torch.mean(centers, dim=1, keepdim=True)
    dist = torch.norm(centers - avg_cam_center, dim=0, keepdim=True)
    diagonal = torch.max(dist).item()
    return avg_cam_center.flatten(), diagonal


def get_nerf_pp_norm(cam_info: list[CameraInfo]):
    cam_centers = [get_world2view(cam.R, cam.T).inverse()[:3, 3] for cam in cam_info]
    center, diagonal = get_center_and_diag(cam_centers)  # type: ignore
    radius = diagonal * 1.1
    translate = -center
    return NerfNormalization(translate=translate, radius=radius)


def fetch_ply(path: Path):
    plydata = PlyData.read(str(path))
    vertices = plydata["vertex"]
    positions = torch.from_numpy(
        np.vstack([vertices["x"], vertices["y"], vertices["z"]])
    ).T
    colors = (
        torch.from_numpy(
            np.vstack([vertices["red"], vertices["green"], vertices["blue"]])
        ).T
        / 255.0
    )
    normals = torch.from_numpy(
        np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]])
    ).T
    is_random = False
    if plydata.comments:
        for comment in plydata.comments:
            if comment.startswith("random="):
                is_random = bool(comment.split("=", 1)[1] == "True")
    return BasicPointCloud(
        points=positions.contiguous(),
        colors=colors.contiguous(),
        normals=normals.contiguous(),
    ), is_random


def store_ply(path: Path, xyz: npt.NDArray, rgb: npt.NDArray, is_random: bool = False):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.comments.append(f"random={is_random}")
    ply_data.write(str(path))


def init_parameters(size: Sequence[int], method: str, device: torch.device):
    if method == "sh":
        return RGB2SH(torch.rand(size, dtype=torch.float32, device=device))
    elif method == "randn":
        return torch.randn(size, dtype=torch.float32, device=device)
    elif method == "rand":
        return torch.rand(size, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def store_camera_infos_as_json(cameras: list[CameraInfo], cameras_path: Path | str):
    cams: list[dict] = []
    for camera in cameras:
        Rt = torch.zeros((4, 4))
        Rt[:3, :3] = camera.R.transpose(0, 1)
        Rt[:3, 3] = camera.T
        Rt[3, 3] = 1.0

        C2W = Rt.inverse()
        T = C2W[:3, 3]
        R = C2W[:3, :3]
        camera_entry = {
            "id": camera.uid,
            "img_name": camera.image_name,
            "img_path": str(camera.image_path),
            "width": camera.width,
            "height": camera.height,
            "position": T.tolist(),
            "rotation": R.tolist(),
            "fx": fov2focal(camera.fovX, camera.width),
            "fy": fov2focal(camera.fovY, camera.height),
        }
        cams.append(camera_entry)

    with open(cameras_path, "w") as file:
        json.dump(cams, file)


def generate_random_point_cloud(
    ply_path: Path,
    num_pts: int,
    scale: float,
    bias: torch.Tensor,
    save: bool = True,
):
    print(f"Generating random point cloud ({num_pts})...")
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = (2 * torch.rand((num_pts, 3)) - 1) * scale + bias
    shs = torch.rand((num_pts, 3)) / 255.0
    colors = SH2RGB(shs)
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=torch.zeros((num_pts, 3)))
    if save:
        store_ply(ply_path, xyz.numpy(), colors.numpy() * 255, True)
    return pcd


def save_scene_info(scene_info: SceneInfo, model_path: Path):
    pcd = scene_info.point_cloud
    if pcd is not None:
        store_ply(
            model_path / "input.ply",
            pcd.points.numpy(),
            pcd.colors.numpy() * 255,
            scene_info.is_pcd_random,
        )

    # Store the camera infos as json
    camlist: list[CameraInfo] = []
    if scene_info.test_cameras:
        camlist.extend(scene_info.test_cameras)
    if scene_info.train_cameras:
        camlist.extend(scene_info.train_cameras)
    store_camera_infos_as_json(camlist, model_path / "cameras.json")
