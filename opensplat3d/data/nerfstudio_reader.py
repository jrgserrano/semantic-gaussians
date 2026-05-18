import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch

from opensplat3d.data.reader import Reader, sample
from opensplat3d.utils.camera_utils import focal2fov
from opensplat3d.utils.scene_utils import CameraInfo


def filter_bad_frames(frames: list[dict[str, Any]]):
    num_bad_frames = sum(x["is_bad"] for x in frames if "is_bad" in x)
    if num_bad_frames > 0:
        frames = [x for x in frames if "is_bad" not in x or not x["is_bad"]]
    return frames, num_bad_frames


class NerfStudioReader(Reader[tuple[int, dict[str, Any]]]):
    def __init__(
        self,
        path: Path,
        image_subdir: str = "images",
        mask_subdir: str | None = None,
        mask_level: str = "default",
        num_frames: int = -1,
        nth_frames: int = -1,
        frames_dist: str = "uniform",
    ):
        self.path = path
        self.images_folder = path / image_subdir
        self.mask_subdir = mask_subdir
        self.mask_level = mask_level

        with open(self.path / "transforms.json") as json_file:
            contents = json.load(json_file)

        assert contents["camera_model"] == "PINHOLE", (
            f"Only PINHOLE cameras supported but got {contents['camera_model']}"
        )

        self.width: int = contents["w"]
        self.height: int = contents["h"]

        focalX: float = contents["fl_x"]
        focalY: float = contents["fl_y"]

        self.fovX = focal2fov(focalX, self.width)
        self.fovY = focal2fov(focalY, self.height)

        train_frames: list[dict[str, Any]] = contents["frames"]
        test_frames: list[dict[str, Any]] = contents["test_frames"]

        num_train_frames = len(train_frames)
        num_test_frames = len(test_frames)
        train_frames, num_bad_train_frames = filter_bad_frames(
            sorted(train_frames, key=lambda x: x["file_path"])
        )
        test_frames, num_bad_test_frames = filter_bad_frames(
            sorted(test_frames, key=lambda x: x["file_path"])
        )
        print(
            f"Removing {num_bad_train_frames}/{num_train_frames} training cameras based on is_bad flag"
        )
        print(
            f"Removing {num_bad_test_frames}/{num_test_frames} testing cameras based on is_bad flag"
        )

        train_keys = list(enumerate(train_frames))
        test_keys = list(enumerate(test_frames))

        train_keys = sample(train_keys, num_frames, nth_frames, frames_dist)

        super().__init__(train_keys, test_keys)

    def read_camera(self, key: tuple[int, dict[str, Any]]) -> CameraInfo:
        idx, frame = key
        fpath = Path(frame["file_path"])
        assert fpath.suffix.lower() in {
            ".jpg",
            ".jpeg",
        }, f"Invalid image path {fpath}"

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = torch.tensor(frame["transform_matrix"])
        # Convert ScanNet++ nerfstudio back to COLMAP: https://github.com/scannetpp/scannetpp/blob/136c416baa915738c27db1aad198429be8fba68d/common/utils/nerfstudio.py#L49
        c2w[2, :] *= -1
        c2w = c2w[[1, 0, 2, 3], :]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = c2w.inverse()

        R: torch.FloatTensor = (
            w2c[:3, :3].transpose(0, 1).float()  # type: ignore
        )  # R is stored transposed due to 'glm' in CUDA code
        T: torch.FloatTensor = w2c[:3, 3]  # type: ignore

        image_path = self.images_folder / fpath
        image: torch.ByteTensor = torch.from_numpy(  # type: ignore
            iio.imread(image_path, pilmode="RGB")
        )  # HWC
        width = image.size(1)
        height = image.size(0)
        assert image.size(2) == 3
        assert width == self.width and height == self.height, (
            f"Image size mismatch: {width}x{height} vs expected {self.width}x{self.height}"
        )

        masks: torch.Tensor | None = None
        if self.mask_subdir is not None:
            masks_path = self.path / self.mask_subdir / (fpath.stem + ".npz")
            assert masks_path.exists(), f"Mask {masks_path} does not exist"
            with np.load(masks_path) as level_masks:
                masks = torch.from_numpy(level_masks[self.mask_level]).long()

        depth: torch.Tensor | None = None
        if "depth_file_path" in frame:
            depth_path = self.path / frame["depth_file_path"]
            if depth_path.exists():
                depth_img = iio.imread(depth_path)
                # Convert from millimeters (uint16) to meters (float32)
                depth = torch.from_numpy(depth_img.astype(np.float32) / 1000.0)

        return CameraInfo(
            uid=idx,
            R=R,
            T=T,
            fovX=self.fovX,
            fovY=self.fovY,
            image=image,
            image_path=image_path,
            image_name=image_path.stem,
            width=width,
            height=height,
            masks=masks,
            depth=depth,
        )
