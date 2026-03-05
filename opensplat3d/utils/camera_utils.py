import math

import torch


def fov2focal(fov: float, pixels: int):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal: float, pixels: int):
    return 2 * math.atan(pixels / (2 * focal))


def get_world2view(R: torch.FloatTensor, t: torch.FloatTensor) -> torch.FloatTensor:
    Rt: torch.FloatTensor = torch.zeros((4, 4), dtype=torch.float)  # type: ignore
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def get_projection_matrix(
    znear: float, zfar: float, fovX: float, fovY: float
) -> torch.FloatTensor:
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P: torch.FloatTensor = torch.zeros((4, 4), dtype=torch.float)  # type: ignore

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P
