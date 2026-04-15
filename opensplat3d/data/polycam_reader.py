"""
Reader for datasets exported with a single transforms.json file that contains:
  - Per-frame or global camera intrinsics (fl_x, fl_y, cx, cy, w, h)
  - Transform matrices (camera-to-world, OpenGL/NeRF convention)
  - Optional depth paths (depth_path per frame)
  - Optional depth_integer_scale global field

Compatible with Polycam, Record3D, and similar iPhone/iPad capture apps.
"""

import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch

from opensplat3d.data.reader import Reader, sample, split_hold
from opensplat3d.utils.camera_utils import focal2fov
from opensplat3d.utils.scene_utils import CameraInfo
from torchvision.transforms.functional import resize, InterpolationMode


class PolycamReader(Reader[tuple[int, dict[str, Any]]]):
    def __init__(
        self,
        path: Path,
        test_hold: float | int = 8,
        mask_subdir: str | None = None,
        mask_level: str = "default",
        num_frames: int = -1,
        nth_frames: int = -1,
        frames_dist: str = "uniform",
    ):
        self.path = path
        self.mask_subdir = mask_subdir
        self.mask_level = mask_level

        with open(path / "transforms.json") as f:
            contents = json.load(f)

        # Global intrinsics (fallback if not per-frame)
        self.global_fl_x: float = contents.get("fl_x", None)
        self.global_fl_y: float = contents.get("fl_y", None)
        self.global_cx: float = contents.get("cx", None)
        self.global_cy: float = contents.get("cy", None)
        self.global_w: int = contents.get("w", None)
        self.global_h: int = contents.get("h", None)

        # Depth scale: converts raw integer pixels → metres
        # Polycam stores depth as uint16 millimetres by default (scale=1 means 1mm/unit)
        self.depth_integer_scale: float = float(contents.get("depth_integer_scale", 1000.0))

        frames: list[dict[str, Any]] = contents["frames"]
        frames = sorted(frames, key=lambda x: x["file_path"])

        all_keys: list[tuple[int, dict]] = list(enumerate(frames))
        all_keys = sample(all_keys, num_frames, nth_frames, frames_dist)

        train_keys, test_keys = split_hold(all_keys, test_hold)
        super().__init__(train_keys, test_keys)

    def _get_intrinsics(self, frame: dict) -> tuple[float, float, int, int]:
        """Return (fovX, fovY, width, height) using per-frame or global intrinsics."""
        fl_x = frame.get("fl_x", self.global_fl_x)
        fl_y = frame.get("fl_y", self.global_fl_y)
        w = int(frame.get("w", self.global_w))
        h = int(frame.get("h", self.global_h))
        assert fl_x is not None and fl_y is not None, (
            "No focal length found in transforms.json (neither global nor per-frame)"
        )
        return focal2fov(fl_x, w), focal2fov(fl_y, h), w, h

    def read_camera(self, key: tuple[int, dict[str, Any]]) -> CameraInfo:
        idx, frame = key

        fovX, fovY, width, height = self._get_intrinsics(frame)

        scaling = 0.25
        new_w, new_h = int(width * scaling), int(height * scaling)

        # Transform matrix: camera-to-world in OpenGL/NeRF convention (Y-up, Z-back)
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)

        # Convert from OpenGL → COLMAP/OpenCV (Y-down, Z-forward)
        c2w[:3, 1:3] *= -1

        w2c = c2w.inverse()
        R: torch.FloatTensor = w2c[:3, :3].transpose(0, 1).float()  # type: ignore
        T: torch.FloatTensor = w2c[:3, 3].float()  # type: ignore

        # --- RGB image ---
        fpath = Path(frame["file_path"])
        # Support paths with or without extension
        for ext in ["", ".png", ".jpg", ".jpeg"]:
            image_path = self.path / (str(fpath) + ext)
            if image_path.exists():
                break
        assert image_path.exists(), f"Image not found: {self.path / str(fpath)}"

        image_np = iio.imread(image_path, pilmode="RGB")
        image = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() # [C, H, W]
        image = torch.nn.functional.interpolate(image, size=(new_h, new_w), mode="bilinear")
        image = image.squeeze(0).permute(1, 2, 0).byte()

        # --- Depth map (optional) ---
        depth: torch.Tensor | None = None
        depth_path_str: str | None = frame.get("depth_path")
        if depth_path_str is not None:
            depth_path = self.path / depth_path_str
            if depth_path.exists():
                depth_np = iio.imread(depth_path).astype(np.float32)
                # Convert raw integer values to metres
                depth = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
                depth = torch.nn.functional.interpolate(depth, size=(new_h, new_w), mode="nearest").long()
                depth = depth.squeeze(0) / self.depth_integer_scale
            else:
                print(f"Warning: depth map not found at {depth_path}")

        # --- SAM masks (optional) ---
        masks: torch.Tensor | None = None
        if self.mask_subdir is not None:
            masks_path = self.path / self.mask_subdir / (fpath.stem + ".npz")
            if masks_path.exists():
                with np.load(masks_path) as level_masks:
                    masks = torch.from_numpy(level_masks[self.mask_level].astype(np.int32)).unsqueeze(0).unsqueeze(0).float() # [1, 1, H, W]
                    masks = torch.nn.functional.interpolate(masks.float(), size=(new_h, new_w), mode="nearest")
                    masks = masks.squeeze().long()
            else:
                print(f"Warning: mask expected but not found at {masks_path}")

        return CameraInfo(
            uid=idx,
            R=R,
            T=T,
            fovX=fovX,
            fovY=fovY,
            image=image,
            image_path=image_path,
            image_name=fpath.stem,
            width=new_w,
            height=new_h,
            masks=masks,
            depth=depth,
        )
