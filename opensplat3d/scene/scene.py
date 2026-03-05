import random

import torch

from opensplat3d.scene.camera import Camera, to_cameras
from opensplat3d.utils.scene_utils import SceneInfo


class Scene:
    def __init__(
        self,
        scene_info: SceneInfo,
        resolution: int,
        data_device: torch.device,
        shuffle: bool = True,
        resolution_scales: list[float] = [1.0],
    ):
        self.train_cameras: dict[float, list[Camera]] = {}
        self.test_cameras: dict[float, list[Camera]] = {}

        self.cameras_extent = scene_info.nerf_normalization.radius

        train_cameras = scene_info.train_cameras.copy()
        test_cameras = scene_info.test_cameras.copy()

        if shuffle:
            random.shuffle(train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(test_cameras)  # Multi-res consistent random shuffling

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = to_cameras(
                train_cameras,
                resolution_scale,
                resolution,
                data_device,
                progbar=True,
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = to_cameras(
                test_cameras,
                resolution_scale,
                resolution,
                data_device,
                progbar=True,
            )

    def get_train_cameras(self, scale: float = 1.0):
        return self.train_cameras[scale]

    def get_test_cameras(self, scale: float = 1.0):
        return self.test_cameras[scale]
