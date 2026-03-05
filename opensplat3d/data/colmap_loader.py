#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import struct
from dataclasses import dataclass
from io import BufferedReader, TextIOWrapper
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch


class CameraModel(NamedTuple):
    id: int
    name: str
    num_params: int


class CameraIntrinsics(NamedTuple):
    id: int
    model: CameraModel
    width: int
    height: int
    params: torch.Tensor


CAMERA_MODELS = {
    CameraModel(id=0, name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(id=1, name="PINHOLE", num_params=4),
    CameraModel(id=2, name="SIMPLE_RADIAL", num_params=4),
    CameraModel(id=3, name="RADIAL", num_params=5),
    CameraModel(id=4, name="OPENCV", num_params=8),
    CameraModel(id=5, name="OPENCV_FISHEYE", num_params=8),
    CameraModel(id=6, name="FULL_OPENCV", num_params=12),
    CameraModel(id=7, name="FOV", num_params=5),
    CameraModel(id=8, name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(id=9, name="RADIAL_FISHEYE", num_params=5),
    CameraModel(id=10, name="THIN_PRISM_FISHEYE", num_params=12),
}

CAMERA_MODEL_IDS = {m.id: m for m in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {m.name: m for m in CAMERA_MODELS}


def qvec2rotmat(qvec: torch.FloatTensor) -> torch.FloatTensor:
    """
    Quaternion (w,x,y,z) to rotation matrix.
    """
    vec = torch.tensor(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )  # type: ignore
    vec: torch.FloatTensor = vec  # typing: ignore
    return vec


def rotmat2qvec(R: npt.NDArray) -> torch.FloatTensor:
    """
    Rotation matrix to quaternion (w,x,y,z).
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        torch.tensor(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = torch.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], torch.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


@dataclass
class ImageInfo:
    id: int
    qvec: torch.FloatTensor
    tvec: torch.FloatTensor
    camera_id: int
    name: str
    xyz: torch.FloatTensor
    point3D_ids: torch.IntTensor

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(
    fid: TextIOWrapper | BufferedReader,
    num_bytes: int,
    format_char_sequence: str,
    endian_character: str = "<",
):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)  # type: ignore


def read_points3D_text(path: Path | str):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors


def read_points3D_binary(path_to_model_file: Path | str):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            _ = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def read_intrinsics_text(path: Path | str, assert_pinhole: bool = True):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    camera_intrinsics: dict[int, CameraIntrinsics] = {}
    with open(path) as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                if assert_pinhole:
                    assert model == "PINHOLE", (
                        f"While the loader support the camera model {model}, the rest of the code assumes PINHOLE"
                    )
                width = int(elems[2])
                height = int(elems[3])
                params = torch.tensor(tuple(map(float, elems[4:])))
                camera_intrinsics[camera_id] = CameraIntrinsics(
                    id=camera_id,
                    model=CAMERA_MODEL_NAMES[model],
                    width=width,
                    height=height,
                    params=params,
                )
    return camera_intrinsics


def read_extrinsics_binary(path_to_model_file: Path | str):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images: dict[int, ImageInfo] = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])  # type: ignore
            tvec = np.array(binary_image_properties[5:8])  # type: ignore
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            elems = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xyz: torch.FloatTensor = torch.column_stack(  # type: ignore
                [
                    torch.tensor(tuple(map(float, elems[0::3]))),
                    torch.tensor(tuple(map(float, elems[1::3]))),
                ]
            )
            point3D_ids: torch.IntTensor = torch.tensor(  # type: ignore
                tuple(map(int, elems[2::3])), dtype=torch.int32
            )
            qvec: torch.FloatTensor = torch.from_numpy(qvec)  # type: ignore
            tvec: torch.FloatTensor = torch.from_numpy(tvec)  # type: ignore
            images[image_id] = ImageInfo(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xyz=xyz,
                point3D_ids=point3D_ids,
            )
    return images


def read_intrinsics_binary(path_to_model_file: Path | str, assert_pinhole: bool = True):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    camera_intrinsics: dict[int, CameraIntrinsics] = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            elems = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = int(elems[0])
            model_id = int(elems[1])
            model = CAMERA_MODEL_IDS[model_id]
            if assert_pinhole:
                assert model.name == "PINHOLE", (
                    "While the loader support other types, the rest of the code assumes PINHOLE"
                )
            width = int(elems[2])
            height = int(elems[3])
            params = read_next_bytes(
                fid,
                num_bytes=8 * model.num_params,
                format_char_sequence="d" * model.num_params,
            )
            camera_intrinsics[camera_id] = CameraIntrinsics(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                params=torch.tensor(tuple(map(float, params))),
            )
        assert len(camera_intrinsics) == num_cameras
    return camera_intrinsics


def read_extrinsics_text(path: Path | str):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images: dict[int, ImageInfo] = {}
    with open(path) as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec: torch.FloatTensor = torch.tensor(tuple(map(float, elems[1:5])))  # type: ignore
                tvec: torch.FloatTensor = torch.tensor(tuple(map(float, elems[5:8])))  # type: ignore
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xyz: torch.FloatTensor = torch.column_stack(  # type: ignore
                    [
                        torch.tensor(tuple(map(float, elems[0::3]))),
                        torch.tensor(tuple(map(float, elems[1::3]))),
                    ]
                )
                point3D_ids: torch.IntTensor = torch.tensor(  # type: ignore
                    tuple(map(int, elems[2::3])), dtype=torch.int32
                )
                images[image_id] = ImageInfo(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xyz=xyz,
                    point3D_ids=point3D_ids,
                )
    return images


def read_colmap_bin_array(path: Path | str):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()
