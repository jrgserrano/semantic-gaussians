import math
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch

from opensplat3d.data.reader import Reader, sample
from opensplat3d.utils.camera_utils import focal2fov, fov2focal
from opensplat3d.utils.scene_utils import CameraInfo


def parse_traj_txt(path: Path) -> list[torch.FloatTensor]:
    """Parse the Replica traj.txt file which contains 4x4 flattened matrices."""
    c2ws = []
    with open(path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 16:
                c2w = torch.tensor(values, dtype=torch.float32).reshape(4, 4)
                c2ws.append(c2w)
    return c2ws


class ReplicaReader(Reader[tuple[int, torch.FloatTensor]]):
    def __init__(
        self,
        path: Path,
        images_subdir: str = "results",
        num_frames: int = -1,
        nth_frames: int = -1,
        frames_dist: str = "uniform",
        mask_subdir: str | None = None,
        mask_level: str = "default",
    ):
        self.path = path
        self.images_subdir = images_subdir
        self.mask_subdir = mask_subdir
        self.mask_level = mask_level

        c2w_matrices = parse_traj_txt(path / "traj.txt")
        # Combine index with c2w matrix for the data loader keys
        keys = list(enumerate(c2w_matrices))

        # Train keys
        train_keys = sample(keys, num_frames, nth_frames, frames_dist)

        # For Replica without a specific split, let's keep test_keys empty or use identical for simplicity if not evaluating holding
        super().__init__(train_keys=train_keys, test_keys=[])

    def read_camera(self, key: tuple[int, torch.FloatTensor]) -> CameraInfo:
        idx, c2w = key
        
        # In Replica, C2W uses OpenCV/OpenGL convention but we usually align it based on the coordinate system required.
        # Often we need to build world-to-cam for gaussian-splatting
        w2c = c2w.inverse()
        R: torch.FloatTensor = (
            w2c[:3, :3].transpose(0, 1).float()
        )
        T: torch.FloatTensor = w2c[:3, 3]

        # Replica images are usually frame000000.jpg
        img_name = f"frame{idx:06d}.jpg"
        image_path = self.path / self.images_subdir / img_name
        
        # Read the RGB image
        if image_path.exists():
            image_np = iio.imread(image_path)
            if image_np.shape[2] == 3:
                # Add alpha channel 255
                alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
                image_np = np.concatenate([image_np, alpha], axis=2)
            image: torch.ByteTensor = torch.from_numpy(image_np)
        else:
            raise FileNotFoundError(f"Missing {image_path}")

        width = image.size(1)
        height = image.size(0)

        # Read the Depth image
        depth_name = f"depth{idx:06d}.png"
        depth_path = self.path / self.images_subdir / depth_name
        depth: torch.Tensor | None = None
        if depth_path.exists():
            depth_np = iio.imread(depth_path)
            # Typically 16-bit PNG depth maps. Assuming scale (mm). Converting to float meters.
            # Some replica depth renders require scaling by focal or division. Using generic scale
            # We scale by 65535 or 1000.0. Let's provide a generic scale approach:
            depth = torch.from_numpy(depth_np.astype(np.float32))
            if depth.max() > 0:
                # Normalizing depth roughly to 0-10 meters based on 65535 bounds
                depth = depth / 6553.5

        # We assume a generic 90 degree FOV for replica dataset if none provided
        cam_fx = 600.0  # Approx for 1200x680
        if width == 1200:
            cam_fx = 600.0
            
        fovX = focal2fov(cam_fx, width)
        fovY = focal2fov(cam_fx, height)

        masks: torch.Tensor | None = None
        if self.mask_subdir is not None:
            masks_path = self.path / self.mask_subdir / f"{image_path.stem}.npz"
            if masks_path.exists():
                with np.load(masks_path) as level_masks:
                    masks = torch.from_numpy(level_masks[self.mask_level]).long()
            else:
                print(f"Warning: Mask expected but not found at {masks_path}")

        return CameraInfo(
            uid=idx,
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
