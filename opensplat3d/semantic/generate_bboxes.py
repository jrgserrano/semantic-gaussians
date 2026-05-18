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
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    unique_labels = unique_labels[unique_labels != -1] # Ignore noise
    
    # Pre-filter non-finite values (NaN, Inf)
    valid_mask = np.all(np.isfinite(xyz), axis=1)
    
    # ALSO filter out points exactly at [0,0,0] which are common artifacts
    origin_mask = np.linalg.norm(xyz, axis=1) < 1e-6
    valid_mask = valid_mask & (~origin_mask)

    if not np.all(valid_mask):
        print(f"Warning: Found {np.sum(~valid_mask)} invalid or origin points. Filtering them out.")
        xyz = xyz[valid_mask]
        labels = labels[valid_mask]

    print(f"Found {len(unique_labels)} valid clusters. Calculating bounding boxes...")
    
    bboxes = {}
    for label in tqdm(unique_labels, desc="Calculating BBoxes"):
        if label_counts.get(label, 0) < 50: # Skip very small clusters that are likely noise
            continue

        mask = (labels == label)
        cluster_xyz = xyz[mask]
        
        if len(cluster_xyz) == 0:
            continue
            
        # Calculate robust AABB using IQR (Interquartile Range) to avoid outliers
        # This is extremely robust against points far away (e.g. at origin or infinity)
        q1 = np.percentile(cluster_xyz, 25, axis=0)
        q3 = np.percentile(cluster_xyz, 75, axis=0)
        iqr = q3 - q1
        
        # Standard outlier detection: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        # We use a slightly more generous 2.0 to avoid clipping real parts of the object
        lower_bound = q1 - 2.0 * iqr
        upper_bound = q3 + 2.0 * iqr
        
        mask_robust = np.all((cluster_xyz >= lower_bound) & (cluster_xyz <= upper_bound), axis=1)
        cluster_xyz_robust = cluster_xyz[mask_robust]
        
        if len(cluster_xyz_robust) > 0:
            bbox_min = np.min(cluster_xyz_robust, axis=0)
            bbox_max = np.max(cluster_xyz_robust, axis=0)
        else:
            # Fallback to 10-90 percentiles if IQR is too restrictive
            bbox_min = np.percentile(cluster_xyz, 15, axis=0)
            bbox_max = np.percentile(cluster_xyz, 85, axis=0)
        
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
