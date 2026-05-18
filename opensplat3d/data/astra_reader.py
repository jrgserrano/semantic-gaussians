
import json
from pathlib import Path
from typing import Any
import cv2
import imageio.v3 as iio
import numpy as np
import torch
from tqdm.contrib.concurrent import thread_map

from opensplat3d.data.reader import Reader
from opensplat3d.utils.camera_utils import focal2fov
from opensplat3d.utils.scene_utils import CameraInfo

class AstraReader(Reader[tuple[int, dict[str, Any]]]):
    def __init__(self, path: Path, **kwargs):
        self.path = path
        json_path = path / "transforms.json"
        if not json_path.exists():
            raise FileNotFoundError(f"No transforms.json found in {path}")
            
        with open(json_path, "r") as f:
            self.meta = json.load(f)
            
        self.width = self.meta["w"]
        self.height = self.meta["h"]
        self.frames = self.meta["frames"]
        
        self.mask_subdir = kwargs.get("mask_subdir", "sam")
        self.mask_level = kwargs.get("mask_level", "default")
        
        # Subsampling
        nth = kwargs.get("nth_frames", 1)
        if nth > 1:
            self.frames = self.frames[::nth]
            
        # FILTER: Añadir rutas de profundidad automáticamente si no están en el JSON
        # Esto hace el Reader compatible con transforms.json generados por hloc/odometría
        depth_dir = self.path / "depth"
        depth_ia_dir = self.path / "depth_ia"
        valid_frames = []
        for frame in self.frames:
            fpath = Path(frame["file_path"])
            stem = fpath.stem
            
            # Si no tiene depth_file_path explícito, intentamos inferirlo
            if "depth_file_path" not in frame:
                inferred_depth = depth_dir / f"{stem}.png"
                if inferred_depth.exists():
                    frame["depth_file_path"] = str(Path("depth") / f"{stem}.png")
            
            # Verificar que la profundidad existe (sea explícita o inferida)
            if "depth_file_path" in frame:
                depth_path = self.path / frame["depth_file_path"]
                if depth_path.exists():
                    valid_frames.append(frame)
        
        if len(valid_frames) < len(self.frames):
            print(f"Filtering: Kept {len(valid_frames)} frames out of {len(self.frames)} (discarded frames without depth)")
        self.frames = valid_frames

        max_f = kwargs.get("num_frames", -1)
        if max_f > 0:
            self.frames = self.frames[:max_f]

    def load_train(self, progbar: bool = False):
        keys = list(enumerate(self.frames))
        desc = "Loading Astra Training Cameras"
        return thread_map(self.read_camera, keys, desc=desc, disable=not progbar)

    def load_test(self, progbar: bool = False):
        return []

    def read_camera(self, data: tuple[int, dict[str, Any]]) -> CameraInfo:
        uid, frame = data
        
        fpath = Path(frame["file_path"])
        image_path = self.path / fpath
        if not image_path.exists():
            image_path = self.path / "images" / fpath.name
            
        image = iio.imread(image_path)
        image = iio.imread(image_path)
        # No swap needed - images are already RGB on disk
        
        # Transform matrix (C2W)
        c2w = np.array(frame["transform_matrix"])
        
        # Get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        
        # Intrinsics
        fl_x = self.meta.get("fl_x", 516.45)
        fl_y = self.meta.get("fl_y", fl_x)
        cx = self.meta.get("cx", self.width / 2)
        cy = self.meta.get("cy", self.height / 2)
        
        fovX = focal2fov(fl_x, self.width)
        fovY = focal2fov(fl_y, self.height)
        
        # Mask loading
        masks = None
        if self.mask_subdir:
            mask_path = self.path / self.mask_subdir / f"{fpath.stem}.npz"
            if mask_path.exists():
                try:
                    mask_data = np.load(mask_path)
                    if self.mask_level in mask_data:
                        masks = mask_data[self.mask_level]
                except Exception as e:
                    print(f"Warning: Could not load mask {mask_path}: {e}")

        # Depth loading (Astra metric)
        depth = None
        if "depth_file_path" in frame:
            depth_rel_path = Path(frame["depth_file_path"])
            depth_path = self.path / depth_rel_path
            if depth_path.exists():
                depth_raw = iio.imread(depth_path)
                depth = depth_raw.astype(np.float32) / 1000.0

        # Normal map generation (from smooth IA depth)
        normal = None
        # Buscamos la profundidad de la IA en la carpeta que creamos
        depth_ia_path = self.path / "depth_ia" / f"{fpath.stem}.png"
        if depth_ia_path.exists():
            depth_ia_raw = iio.imread(depth_ia_path)
            depth_ia = depth_ia_raw.astype(np.float32) / 1000.0
            normal = self._compute_normals(depth_ia, fl_x, fl_y)

        return CameraInfo(
            uid=uid,
            R=torch.from_numpy(R).float(),
            T=torch.from_numpy(T).float(),
            fovX=fovX,
            fovY=fovY,
            image=torch.from_numpy(image), # ByteTensor (uint8)
            image_path=image_path,
            image_name=fpath.stem,
            width=self.width,
            height=self.height,
            masks=torch.from_numpy(masks).long() if masks is not None else None,
            depth=torch.from_numpy(depth).float() if depth is not None else None,
            cx=cx,
            cy=cy,
            normal=torch.from_numpy(normal).float() if normal is not None else None
        )

    def _compute_normals(self, depth, fx, fy):
        """Calcula mapa de normales a partir de profundidad usando NumPy/CV2."""
        # depth: (H, W) en metros
        # Usamos Sobel para gradientes suaves
        dz_dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        dz_dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        
        # La normal en un punto es proporcional a (-dz/dx * fx/z, -dz/dy * fy/z, 1)
        normal = np.dstack((-dz_dx * fx / (depth + 1e-6), 
                           -dz_dy * fy / (depth + 1e-6), 
                           np.ones_like(depth)))
        
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (norm + 1e-6)
        return normal.transpose(2, 0, 1) # Formato CHW
