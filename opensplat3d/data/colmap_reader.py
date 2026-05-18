from operator import itemgetter
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from opensplat3d.data.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from opensplat3d.data.reader import Reader, sample, split_hold
from opensplat3d.utils.camera_utils import focal2fov
from opensplat3d.utils.mask_utils import mask_consecutive_labels
from opensplat3d.utils.scene_utils import CameraInfo


class ColmapReader(Reader[int]):
    def __init__(
        self,
        path: Path,
        image_subdir: str = "images",
        test_hold: float | int = 8,
        mask_subdir: str | None = None,
        mask_level: str = "default",
        num_frames: int = -1,
        nth_frames: int = -1,
        frames_dist: str = "uniform",
    ):
        cameras_extrinsic_file = path / "sparse" / "0" / "images.bin"
        cameras_intrinsic_file = path / "sparse" / "0" / "cameras.bin"
        try:
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except FileNotFoundError:
            cam_extrinsics = read_extrinsics_text(
                cameras_extrinsic_file.with_suffix(".txt")
            )
            cam_intrinsics = read_intrinsics_text(
                cameras_intrinsic_file.with_suffix(".txt")
            )

        self.cam_extrinsics = cam_extrinsics
        self.cam_intrinsics = cam_intrinsics
        self.images_folder = path / image_subdir
        self.mask_subdir = mask_subdir
        self.mask_level = mask_level

        cameras = [(key, Path(x.name).stem) for key, x in cam_extrinsics.items()]
        cameras = sorted(cameras, key=itemgetter(1))

        lerf_ovs_test_image_names = []
        if (self.images_folder.parent.parent / "label").exists():
            print("Detected lerf-ovs format.")
            lerf_ovs_label_path = (
                self.images_folder.parent.parent
                / "label"
                / self.images_folder.parent.name
            )
            assert lerf_ovs_label_path.exists(), (
                f"Expected label directory at {lerf_ovs_label_path}"
            )
            lerf_ovs_test_image_names = sorted(
                [x.stem for x in lerf_ovs_label_path.glob("*.jpg")]
            )

        train_keys = [
            x[0]
            for x in cameras
            if not (x[1].startswith("test_") or x[1] in lerf_ovs_test_image_names)
        ]
        test_keys = [
            x[0]
            for x in cameras
            if x[1].startswith("test_") or x[1] in lerf_ovs_test_image_names
        ]
        if len(test_keys):
            print(
                f"Detected {'lerf-mask' if len(lerf_ovs_test_image_names) == 0 else 'lerf-ovs'} format."
            )
        else:
            # Normal COLMAP format
            train_keys, test_keys = split_hold([x[0] for x in cameras], test_hold)

        train_keys = sample(train_keys, num_frames, nth_frames, frames_dist)

        super().__init__(train_keys, test_keys)

    def read_camera(self, key: int) -> CameraInfo:
        extr = self.cam_extrinsics[key]
        intr = self.cam_intrinsics[extr.camera_id]

        image_path = self.images_folder / Path(extr.name).name
        height = intr.height
        width = intr.width

        # uid = intr.id  # equals extr.camera_id
        R: torch.FloatTensor = qvec2rotmat(extr.qvec).transpose(0, 1)  # type: ignore
        T: torch.FloatTensor = extr.tvec  # type: ignore

        if intr.model.name == "SIMPLE_PINHOLE":
            focal_length_x: float = intr.params[0].item()
            fovY = focal2fov(focal_length_x, height)
            fovX = focal2fov(focal_length_x, width)
        elif intr.model.name == "PINHOLE":
            focal_length_x: float = intr.params[0].item()
            focal_length_y: float = intr.params[1].item()
            fovY = focal2fov(focal_length_y, height)
            fovX = focal2fov(focal_length_x, width)
        else:
            assert False, (
                f"Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported, but got {intr.model.name}!"
            )

        image: torch.ByteTensor = torch.from_numpy(iio.imread(image_path))  # type: ignore # HWC

        masks: torch.Tensor | None = None
        if self.mask_subdir is not None:
            if self.mask_subdir == "object_mask":
                masks_path = (
                    image_path.parent.parent
                    / self.mask_subdir
                    / f"{image_path.stem}.png"
                )
                assert masks_path.exists(), f"Mask {masks_path} does not exist"
                masks = mask_consecutive_labels(
                    torch.from_numpy(iio.imread(masks_path, pilmode="L")).long()
                )
            else:
                masks_path = (
                    image_path.parent.parent
                    / self.mask_subdir
                    / f"{image_path.stem}.npz"
                )
                assert masks_path.exists(), f"Mask {masks_path} does not exist"
                with np.load(masks_path) as level_masks:
                    masks = torch.from_numpy(level_masks[self.mask_level]).long()

        depth: torch.Tensor | None = None
        depth_path = image_path.parent.parent / "depth" / f"{image_path.stem}.png"
        if depth_path.exists():
            depth_np = iio.imread(depth_path).astype(np.float32)
            # Normalize from uint16 (0-65535) to a relative range. 
            # Note: For monocular depth, scale is relative.
            depth = torch.from_numpy(depth_np).unsqueeze(0) / 65535.0

        return CameraInfo(
            uid=extr.id,
            R=R,
            T=T,
            fovX=fovX,
            fovY=fovY,
            image=image,
            image_path=image_path,
            image_name=image_path.stem,
            width=width,
            height=height,
            masks=masks,
            depth=depth,
        )
