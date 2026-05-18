import numpy as np
import torch
from pathlib import Path
import imageio.v3 as iio
from opensplat3d.data.reader import Reader, sample, split_hold
from opensplat3d.utils.camera_utils import focal2fov
from opensplat3d.utils.scene_utils import CameraInfo

def load_matrix(path: Path) -> torch.FloatTensor:
    return torch.from_numpy(np.loadtxt(path)).float()

class ScanNetReader(Reader[int]):
    def __init__(
        self,
        path: Path,
        test_hold: float | int = 8,
        num_frames: int = -1,
        nth_frames: int = -1,
        frames_dist: str = "uniform",
        mask_subdir: str | None = None,
        mask_level: str = "default",
        **kwargs
    ):
        self.path = path
        self.mask_subdir = mask_subdir
        self.mask_level = mask_level
        
        # Load color intrinsics
        intrinsic_path = path / "intrinsic" / "intrinsic_color.txt"
        if not intrinsic_path.exists():
            # Try to find any intrinsic file if the naming is different
            intrinsic_files = list((path / "intrinsic").glob("*.txt"))
            if intrinsic_files:
                intrinsic_path = intrinsic_files[0]
            else:
                raise FileNotFoundError(f"Could not find intrinsic file in {path / 'intrinsic'}")
        
        self.intrinsic = load_matrix(intrinsic_path)
        
        # Find all color images to get indices
        color_path = path / "color"
        image_files = sorted(color_path.glob("*.jpg"), key=lambda x: int(x.stem))
        indices = [int(f.stem) for f in image_files]
        
        # Filter indices based on available poses and valid poses (no inf)
        pose_path = path / "pose"
        valid_indices = []
        for idx in indices:
            p_file = pose_path / f"{idx}.txt"
            if p_file.exists():
                pose = load_matrix(p_file)
                if not torch.any(torch.isinf(pose)) and not torch.any(torch.isnan(pose)):
                    valid_indices.append(idx)
        
        print(f"Found {len(valid_indices)} valid frames with poses out of {len(indices)} images.")
        
        all_keys = sample(valid_indices, num_frames, nth_frames, frames_dist)
        train_keys, test_keys = split_hold(all_keys, test_hold)
        super().__init__(train_keys=train_keys, test_keys=test_keys)

    def read_camera(self, idx: int) -> CameraInfo:
        # Load Pose (C2W)
        c2w = load_matrix(self.path / "pose" / f"{idx}.txt")
        
        # Invert to get W2C
        w2c = torch.inverse(c2w)
        
        # R in CameraInfo is expected to be the transpose of the W2C rotation (which is C2W rotation)
        # T in CameraInfo is expected to be the W2C translation
        R = c2w[:3, :3]
        T = w2c[:3, 3]
        
        # Load Image
        image_path = self.path / "color" / f"{idx}.jpg"
        image_np = iio.imread(image_path)
        
        # Ensure RGBA
        if image_np.shape[2] == 3:
            alpha = np.full((image_np.shape[0], image_np.shape[1], 1), 255, dtype=np.uint8)
            image_np = np.concatenate([image_np, alpha], axis=2)
        image = torch.from_numpy(image_np)
        
        width = image.size(1)
        height = image.size(0)
        
        # Intrinsics
        fx = self.intrinsic[0, 0]
        fy = self.intrinsic[1, 1]
        cx = self.intrinsic[0, 2]
        cy = self.intrinsic[1, 2]
        
        fovX = focal2fov(fx, width)
        fovY = focal2fov(fy, height)
        
        # Depth
        depth_path = self.path / "depth" / f"{idx}.png"
        depth = None
        if depth_path.exists():
            depth_np = iio.imread(depth_path).astype(np.float32)
            # ScanNet depth is in mm, convert to meters
            depth = torch.from_numpy(depth_np / 1000.0)
            # If depth resolution is different from image resolution, it will be handled by the trainer/renderer
            # but we can optionally resize here. However, CameraInfo doesn't mandate matching resolution.
            
        # Masks
        masks = None
        if self.mask_subdir is not None:
             masks_path = self.path / self.mask_subdir / f"{idx}.npz"
             if not masks_path.exists():
                 # Try .png if .npz doesn't exist (some pipelines use png)
                 masks_path = self.path / self.mask_subdir / f"{idx}.png"
             
             if masks_path.exists():
                 if masks_path.suffix == ".npz":
                     with np.load(masks_path) as level_masks:
                         masks = torch.from_numpy(level_masks[self.mask_level]).long()
                 else:
                     masks_np = iio.imread(masks_path)
                     masks = torch.from_numpy(masks_np.astype(np.int64))

        return CameraInfo(
            uid=idx,
            R=R,
            T=T,
            fovX=fovX,
            fovY=fovY,
            image=image,
            image_path=image_path,
            image_name=str(idx),
            width=width,
            height=height,
            masks=masks,
            depth=depth,
            cx=cx,
            cy=cy
        )
