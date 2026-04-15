import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from opensplat3d.utils.setup_utils import setup_model

def generate_bboxes(model_path: Path):
    # Setup model and gaussians
    print(f"Loading model from {model_path}...")
    setup_params = setup_model(model_path)
    gaussians = setup_params.gaussians
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    
    # Load labels
    labels_path = model_path / "clustering" / "labels.npy"
    if not labels_path.exists():
        print(f"Error: Labels file not found at {labels_path}")
        return
    
    labels = np.load(labels_path)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1] # Ignore noise
    
    print(f"Found {len(unique_labels)} valid clusters. Calculating bounding boxes...")
    
    bboxes = {}
    for label in tqdm(unique_labels, desc="Calculating BBoxes"):
        mask = (labels == label)
        cluster_xyz = xyz[mask]
        
        if len(cluster_xyz) == 0:
            continue
            
        # Calculate robust AABB using percentiles to avoid outliers
        bbox_min = np.percentile(cluster_xyz, 10, axis=0)
        bbox_max = np.percentile(cluster_xyz, 90, axis=0)
        
        bboxes[int(label)] = {
            "min": bbox_min.tolist(),
            "max": bbox_max.tolist(),
            "center": ((bbox_min + bbox_max) / 2).tolist(),
            "size": (bbox_max - bbox_min).tolist()
        }
    
    # Save results
    output_path = model_path / "clustering" / "bboxes.pth"
    torch.save(bboxes, output_path)
    print(f"Saved {len(bboxes)} bounding boxes to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bounding boxes for Gaussian clusters")
    parser.add_argument("model_dir", type=Path, help="Path to the model directory")
    args = parser.parse_args()
    
    generate_bboxes(args.model_dir)
