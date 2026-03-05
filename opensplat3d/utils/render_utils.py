"""
Copied from convex-splatting.
"""

import copy
import math
from typing import Literal

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from opensplat3d.scene.camera import Camera, ViewerCamera, get_projection_matrix


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def transform_poses_pca(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transforms poses so principal components lie on XYZ axes.

    Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform


def generate_ellipse_path(
    poses: np.ndarray,
    num_frames: int = 120,
    const_speed: bool = True,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                z_variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)
                ),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, num_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:

    # # Resample theta angles so that the velocity is closer to constant.
    # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    # positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_path(
    viewpoint_cameras: list[Camera],
    num_frames: int = 480,
    z_variation: float = 0.0,
    z_phase: float = 0.0,
) -> list[Camera]:
    c2ws = np.array(
        [
            np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy()))
            for cam in viewpoint_cameras
        ]
    )
    pose = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

    # generate new poses
    new_poses = generate_ellipse_path(
        poses=pose_recenter,
        num_frames=num_frames,
        z_variation=z_variation,
        z_phase=z_phase,
    )
    # warp back to orignal scale
    new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

    traj: list[Camera] = []
    for c2w in new_poses:
        c2w = c2w @ np.diag([1, -1, -1, 1])
        cam = copy.deepcopy(viewpoint_cameras[0])
        cam.image_height = int(cam.image_height / 2) * 2
        cam.image_width = int(cam.image_width / 2) * 2
        cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
        cam.full_proj_transform = (
            cam.world_view_transform.unsqueeze(0).bmm(
                cam.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        traj.append(cam)

    return traj


# Custom interpolation function for smooth camera trajectory along given cameras.
def compute_cumulative_distances(points: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1),
            torch.cumsum(torch.norm(points[1:] - points[:-1], dim=1), dim=0),
        ]
    )


def get_interpolation_times(
    num: int, distances: torch.Tensor, timing: str
) -> torch.Tensor:
    if timing == "uniform":
        return torch.linspace(0, 1, steps=num)
    else:
        # Normalize cumulative distances to [0, 1]
        norm_distances = distances / distances[-1]
        # Times at keyframes range uniformly [0,1]
        num_keys = len(norm_distances)
        key_times = torch.linspace(0, 1, steps=num_keys)
        # Target fractions along the path for output frames
        target_fractions = torch.linspace(0, 1, steps=num)
        # Interpolate to find corresponding times
        times_np = np.interp(
            target_fractions.numpy(), norm_distances.numpy(), key_times.numpy()
        )
        return torch.from_numpy(times_np)


def catmull_rom(p0, p1, p2, p3, t):
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
    )


def interpolate_linear_positions(
    positions: torch.Tensor, times: torch.Tensor
) -> torch.FloatTensor:
    total = len(positions) - 1
    indices = times * total
    i0 = torch.floor(indices).long()
    i1 = torch.clamp(i0 + 1, max=total)
    frac = (indices - i0).unsqueeze(1)
    return (1 - frac) * positions[i0] + frac * positions[i1]  # type: ignore


def interpolate_catmull_rom_positions(
    positions: torch.Tensor, times: torch.Tensor
) -> torch.FloatTensor:
    padded = torch.cat([positions[0:1], positions, positions[-1:]], dim=0)
    total = len(positions) - 1
    indices = times * total
    i = torch.floor(indices).long() + 1
    t = (indices - (i - 1)).unsqueeze(1)
    return catmull_rom(padded[i - 1], padded[i], padded[i + 1], padded[i + 2], t)  # type: ignore


def interpolate_rotations(
    rotations: torch.Tensor, times: torch.Tensor
) -> list[torch.FloatTensor]:
    rots = Rotation.from_matrix([r.cpu().numpy() for r in rotations])
    key_times = np.linspace(0, 1, num=len(rots))
    slerp = Slerp(key_times, rots)
    interp_rots = slerp(times.numpy())
    return [torch.tensor(r.as_matrix(), dtype=torch.float32) for r in interp_rots]  # type: ignore


def interpolate_camera_trajectory(
    cameras: list[Camera],
    num_frames: int,
    interpolation: Literal["linear", "catmull-rom"] = "linear",
    timing: Literal["uniform", "distance"] = "uniform",
) -> list[Camera]:
    positions = torch.stack([cam.T for cam in cameras])
    rotations = torch.stack([cam.R for cam in cameras])

    distances = compute_cumulative_distances(positions)
    times = get_interpolation_times(num_frames, distances, timing)

    if interpolation == "linear":
        interp_T = interpolate_linear_positions(positions, times)
    elif interpolation == "catmull-rom":
        interp_T = interpolate_catmull_rom_positions(positions, times)
    else:
        raise ValueError("Unsupported interpolation type")

    interp_R = interpolate_rotations(rotations, times)

    base = cameras[0]
    device = base.world_view_transform.device
    interpolated: list[Camera] = []
    for i in range(num_frames):
        cam = Camera(
            uid=i,
            name=f"interp_{i}",
            R=interp_R[i],
            T=interp_T[i],  # type: ignore
            fovX=base.fovX,
            fovY=base.fovY,
            image=base.original_image,  # type: ignore
            alpha_mask=None,
            masks=None,
        ).to(device)
        interpolated.append(cam)

    return interpolated


def look_at(
    camera_pos: torch.Tensor,
    target_pos: torch.Tensor,
    up=torch.tensor([0, 1, 0], dtype=torch.float32),
) -> torch.Tensor:
    """
    Computes a world2view matrix (4x4) for a camera looking from camera_pos to target_pos.
    'up' is the up vector in world space.
    Returns a 4x4 torch float tensor (world2view transform).
    """
    device = camera_pos.device
    up = up.to(device=device)
    forward = target_pos - camera_pos
    forward = forward / torch.norm(forward)

    right = torch.cross(up, forward, dim=-1)
    right = right / torch.norm(right)

    true_up = torch.cross(forward, right, dim=-1)
    true_up = true_up / torch.norm(true_up)

    R = torch.eye(4, dtype=torch.float32, device=device)
    R[0, :3] = right
    R[1, :3] = true_up
    R[2, :3] = forward

    T = torch.eye(4, dtype=torch.float32, device=device)
    T[:3, 3] = -camera_pos

    # world2view = R * T
    world2view = R @ T
    return world2view


def get_circle_path(
    xyz: torch.Tensor,  # (N, 3) point cloud tensor
    num_frames: int,  # number of cameras on circle
    elevation: float,  # vertical offset from center (y axis)
    base_cam: Camera
    | ViewerCamera,  # example Camera or parameters for FOV, image size, znear, zfar
    distance: float,  # radius of circle around center
) -> list:
    # Compute center of point cloud
    center = xyz.mean(dim=0)

    cameras = []
    for i in range(num_frames):
        theta = 2 * math.pi * i / num_frames
        cam_x = center[0] + distance * math.cos(theta)
        cam_y = center[1] + elevation
        cam_z = center[2] + distance * math.sin(theta)
        cam_pos = torch.tensor(
            [cam_x, cam_y, cam_z], dtype=torch.float32, device=center.device
        )

        # Compute world2view transform (look at center)
        world_view = look_at(cam_pos, center)

        # Transpose world_view for consistency with Camera (column-major)
        world_view_t = world_view.transpose(0, 1)

        # projection matrix from base_cam parameters
        proj = (
            get_projection_matrix(
                znear=base_cam.znear,
                zfar=base_cam.zfar,
                fovX=base_cam.fovX,
                fovY=base_cam.fovY,
            )
            .transpose(0, 1)
            .to(device=center.device)
        )

        full_proj = (
            world_view_t.unsqueeze(0)
            .bmm(proj.unsqueeze(0))
            .squeeze(0)
            .to(device=center.device)
        )

        viewer_cam = ViewerCamera(
            width=base_cam.image_width,
            height=base_cam.image_height,
            fovX=base_cam.fovX,
            fovY=base_cam.fovY,
            znear=base_cam.znear,
            zfar=base_cam.zfar,
            world_view_transform=world_view_t,
            full_proj_transform=full_proj,
        )
        cameras.append(viewer_cam)

    return cameras
