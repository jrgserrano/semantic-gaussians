import torch
import numpy as np
from tqdm import tqdm
from opensplat3d.eval.metrics import calculate_iou, calculate_biou
from opensplat3d.gaussian_renderer import render
from opensplat3d.params import PipeParams

@torch.no_grad()
def evaluate_replica_instance_metrics(gaussians, scene, cameras, labels, device):
    """
    Evaluate mIoU and mBIoU for Replica instance masks.
    Matches predicted labels (from HDBSCAN) to GT masks.
    """
    if len(cameras) == 0:
        return 0.0, 0.0
    
    # Render label images
    # We use a trick: render "color" where each point's color is its label index
    # (normalized or just as integer if rasterizer supports it).
    # Since rasterizer handles floats, we can't easily do discrete labels with alpha blending.
    # BEST APPROACH: Use the point features and GT masks to see if they are separable,
    # OR: Render discrete label by taking the label of the most contributing Gaussian?
    # The rasterizer doesn't expose the 'argmax' Gaussian.
    
    # Standard evaluation for unsupervised segment:
    # 1. Render features.
    # 2. Assign each pixel to a cluster based on predicted_labels of Gaussians.
    # 3. Match clusters to GT instances.

    # Simplified approach for training metrics:
    # For each camera, we render the "label" by choosing the label of the Gaussian
    # with the highest contribution (weight) to that pixel.
    # Since we don't have that, we'll render the labels as 'colors' (1D) and take the nearest.
    
    all_ious = []
    all_bious = []
    
    for cam in tqdm(cameras, desc="Evaluating Instance Metrics"):
        if cam.masks is None:
            continue
            
        # We need a way to render labels. 
        # Trick: Define a 'color' that is actually the label ID.
        # To avoid blending issues, we can use very high opacities or just accept some blending.
        # But a better way is to use the existing clustering.
        
        # Render features (this is what the clustering was based on)
        render_pkg = render(
            cam, gaussians, PipeParams(), 
            torch.zeros(3, device=device), 
            gaussians.active_sh_degree, 
            render_features=True
        )
        rendered_features = render_pkg.features.permute(1, 2, 0) # [H, W, C]
        gt_masks = cam.masks.to(device) # [H, W]
        
        # Classification: For every pixel, find the Gaussian that 'dominates'.
        # Lacking 'argmax', we classify the rendered FEATURE against the mean features of labels.
        unique_labels = torch.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        if len(unique_labels) == 0:
            all_ious.append(0.0)
            all_bious.append(0.0)
            continue
            
        # Calculate mean features for each cluster (one-time or per-frame?)
        # Let's do it globally for consistency.
        cluster_prototypes = []
        for l in unique_labels:
            mask = labels == l
            proto = gaussians.get_features[mask, :rendered_features.shape[-1]].mean(dim=0)
            cluster_prototypes.append(proto)
        cluster_prototypes = torch.stack(cluster_prototypes) # [N_clusters, C]
        
        # Classify pixels
        H, W, C = rendered_features.shape
        flat_features = rendered_features.reshape(-1, C)
        # Distances to prototypes
        dists = torch.cdist(flat_features.unsqueeze(0), cluster_prototypes.unsqueeze(0)).squeeze(0) # [H*W, N_clusters]
        pred_labels_idx = torch.argmin(dists, dim=1)
        pred_labels = unique_labels[pred_labels_idx].reshape(H, W)
        
        # Matching predicted clusters to GT instances
        gt_instances = torch.unique(gt_masks)
        gt_instances = gt_instances[gt_instances != -1] # usually -1 or 0 is background/ignore
        
        if len(gt_instances) == 0:
            continue
            
        frame_ious = []
        frame_bious = []
        
        for gt_id in gt_instances:
            gt_mask_bin = (gt_masks == gt_id).cpu().numpy()
            
            # Find best matching cluster
            best_iou = 0.0
            best_biou = 0.0
            
            # Optimization: only check clusters that overlap with the GT mask
            overlapping_clusters = torch.unique(pred_labels[gt_masks == gt_id])
            for pred_id in overlapping_clusters:
                if pred_id == -1: continue
                pred_mask_bin = (pred_labels == pred_id).cpu().numpy()
                
                iou = calculate_iou(gt_mask_bin, pred_mask_bin)
                if iou > best_iou:
                    best_iou = iou
                    # BIoU is slower, only compute for best IoU candidate?
                    # Or at least check if IoU is reasonable
                    if iou > 0.1:
                        best_biou = calculate_biou(gt_mask_bin, pred_mask_bin)
            
            frame_ious.append(best_iou)
            frame_bious.append(best_biou)
            
        all_ious.append(np.mean(frame_ious))
        all_bious.append(np.mean(frame_bious))
        
    return np.mean(all_ious) if all_ious else 0.0, np.mean(all_bious) if all_bious else 0.0
