import os
import torch
import torchvision
from pathlib import Path
import json

from core.utils.setup_utils import setup
from core.gaussian_renderer import render
from core.params import PipeParams

def main():
    stage1_dir = Path("outputs/Replica/office4_stages")
    stage1_runs = sorted([p for p in stage1_dir.iterdir() if p.is_dir()])
    if not stage1_runs:
        print("No stage1 runs found!")
        return
    stage1_model = stage1_runs[-1]
    
    stage3_model = Path("outputs/Replica/office4_dense_train/20260613170349-f39d30b7")
    
    out_dir = Path("outputs/plots/training_stages")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pipe_params = PipeParams()
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    cam_idx = 10
    
    stages = [
        {"name": "iter_00001", "model": stage1_model, "iter": 1},
        {"name": "iter_00100", "model": stage1_model, "iter": 100},
        {"name": "iter_01000", "model": stage1_model, "iter": 1000},
        {"name": "iter_30000", "model": stage3_model, "iter": 30000},
    ]
    
    for stage in stages:
        print(f"Loading {stage['name']}...")
        setup_params = setup(stage['model'], load_masks=False, iteration=stage['iter'])
        scene = setup_params.scene
        gaussians = setup_params.gaussians
        model_params = setup_params.model_params
        
        # Get cameras
        if hasattr(scene, "get_train_cameras"):
            cameras = scene.get_train_cameras()
        else:
            cameras = scene.train_cameras[1.0]
            
        cam = cameras[cam_idx]
        
        print(f"Rendering {stage['name']}...")
        render_pkg = render(cam, gaussians, pipe_params, background, model_params.sh_degree)
        
        features = render_pkg.features
        if features is not None:
            # Run PCA on features
            from core.utils.vis_utils import feature_image_pca_3d
            from PIL import Image
            pca_img = feature_image_pca_3d(features.detach().cpu().numpy())
            
            out_path = out_dir / f"stage_{stage['name']}_features.jpg"
            Image.fromarray(pca_img).save(out_path)
            print(f"Saved: {out_path}")
        else:
            print(f"Warning: No features found in {stage['name']}")

if __name__ == "__main__":
    main()
