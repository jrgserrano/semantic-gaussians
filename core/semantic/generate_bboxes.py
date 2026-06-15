import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from core.utils.setup_utils import setup_model

def generate_bboxes(model_path: Path):
    print(f"Loading model from {model_path}...")
    setup_params = setup_model(model_path)
    gaussians = setup_params.gaussians
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    
    labels_path = model_path / "clustering" / "labels.npy"
    if not labels_path.exists():
        print(f"Error: Labels file not found at {labels_path}")
        return
    
    labels = np.load(labels_path)
    output_path = model_path / "clustering" / "bboxes.pth"
    generate_bboxes_from_data(xyz, labels, output_path)

def generate_bboxes_from_data(xyz: np.ndarray, labels: np.ndarray, output_path: Path):
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    unique_labels = unique_labels[unique_labels != -1]
    
    valid_mask = np.all(np.isfinite(xyz), axis=1)
    
    origin_mask = np.linalg.norm(xyz, axis=1) < 1e-6
    valid_mask = valid_mask & (~origin_mask)

    if not np.all(valid_mask):
        print(f"Warning: Found {np.sum(~valid_mask)} invalid or origin points. Filtering them out.")
        xyz = xyz[valid_mask]
        labels = labels[valid_mask]

    print(f"Found {len(unique_labels)} valid clusters. Calculating bounding boxes...")
    
    bboxes = {}
    for label in tqdm(unique_labels, desc="Calculating BBoxes"):
        if label_counts.get(label, 0) < 50:
            continue

        mask = (labels == label)
        cluster_xyz = xyz[mask]
        
        if len(cluster_xyz) == 0:
            continue
            
        q1 = np.percentile(cluster_xyz, 25, axis=0)
        q3 = np.percentile(cluster_xyz, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - 2.0 * iqr
        upper_bound = q3 + 2.0 * iqr
        
        mask_robust = np.all((cluster_xyz >= lower_bound) & (cluster_xyz <= upper_bound), axis=1)
        cluster_xyz_robust = cluster_xyz[mask_robust]
        
        if len(cluster_xyz_robust) > 0:
            bbox_min = np.min(cluster_xyz_robust, axis=0)
            bbox_max = np.max(cluster_xyz_robust, axis=0)
        else:
            bbox_min = np.percentile(cluster_xyz, 15, axis=0)
            bbox_max = np.percentile(cluster_xyz, 85, axis=0)
        
        bboxes[int(label)] = {
            "min": bbox_min.tolist(),
            "max": bbox_max.tolist(),
            "center": ((bbox_min + bbox_max) / 2).tolist(),
            "size": (bbox_max - bbox_min).tolist()
        }
    
    if output_path is not None:
        torch.save(bboxes, output_path)
        print(f"Saved {len(bboxes)} bounding boxes to {output_path}")
    
    return bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bounding boxes for Gaussian clusters")
    parser.add_argument("model_dir", type=Path, help="Path to the model directory")
    args = parser.parse_args()
    
    generate_bboxes(args.model_dir)
