import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch

from core.data.reader import Reader, sample
from core.utils.camera_utils import focal2fov, fov2focal
from core.utils.scene_utils import CameraInfo


def read_transforms(
    path: Path,
) -> tuple[list[tuple[int, dict[str, Any]]], float]:
    with open(path) as json_file:
        contents = json.load(json_file)
        fovX: float = contents["camera_angle_x"]
        # fovY: float = contents.get("camera_angle_y", fovX)
        frames: list[tuple[int, dict[str, Any]]] = list(enumerate(contents["frames"]))
    return frames, fovX


class BlenderReader(Reader[tuple[int, dict[str, Any]]]):
    def __init__(
        self,
        path: Path,
        white_background: bool,
        extension: str = ".png",
        mask_subdir: str | None = None,
        mask_level: str = "default",
        num_frames: int = -1,
        nth_frames: int = -1,
        frames_dist: str = "uniform",
    ):
        self.path = path
        self.white_background = white_background
        self.extension = extension
        self.mask_subdir = mask_subdir
        self.mask_level = mask_level

        train_keys, fovX = read_transforms(path / "transforms_train.json")
        test_keys, test_fovX = read_transforms(path / "transforms_test.json")

        assert fovX == test_fovX, (
            f"FOVs of train ({fovX}) and test ({test_fovX}) sets do not match."
        )
        self.fovX = fovX

        train_keys = sample(train_keys, num_frames, nth_frames, frames_dist)

        super().__init__(train_keys, test_keys)

    def read_camera(self, key: tuple[int, dict[str, Any]]) -> CameraInfo:
        idx, frame = key
        fpath: str = frame["file_path"]
        if not Path(fpath).suffix:
            cam_name = self.path / (fpath + self.extension)
        else:
            cam_name = self.path / fpath
            fpath = str(Path(fpath).with_suffix(""))

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = torch.tensor(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = c2w.inverse()
        R: torch.FloatTensor = (
            w2c[:3, :3].transpose(0, 1).float()  # type: ignore
        )  # R is stored transposed due to 'glm' in CUDA code
        T: torch.FloatTensor = w2c[:3, 3]  # type: ignore

        image_path = self.path / cam_name
        image: torch.ByteTensor = torch.from_numpy(  # type: ignore
            iio.imread(image_path, pilmode="RGBA")
        )  # HWC
        width = image.size(1)
        height = image.size(0)
        assert image.size(2) == 4

        bg = (
            torch.tensor([1, 1, 1])
            if self.white_background
            else torch.tensor([0, 0, 0])
        )

        norm_data = image / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
            1 - norm_data[:, :, 3:4]
        )
        image: torch.ByteTensor = (arr * 255.0).to(dtype=torch.uint8)  # type: ignore

        masks: torch.Tensor | None = None
        if self.mask_subdir is not None:
            masks_path = self.path / self.mask_subdir / (fpath + ".npz")
            if masks_path.parent.name == "color":
                # ScanNet dataset
                masks_path = masks_path.parent.parent / masks_path.name
            assert masks_path.exists(), f"Mask {masks_path} does not exist"
            with np.load(masks_path) as level_masks:
                masks = torch.from_numpy(level_masks[self.mask_level]).long()

        fovY = focal2fov(fov2focal(self.fovX, width), height)

        return CameraInfo(
            uid=idx,
            R=R,
            T=T,
            fovX=self.fovX,
            fovY=fovY,
            image=image,
            image_path=image_path,
            image_name=image_path.stem,
            width=width,
            height=height,
            masks=masks,
        )
